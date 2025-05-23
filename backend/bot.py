import datetime
import io
import asyncio
import os
import sys
import wave
import aiofiles
import json
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferWindowMemory

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import existing services
from db_service import MongoDBService
import call_state

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", colorize=True)

# Global MongoDB service instance
db_service = MongoDBService()

# Define the state for our LangGraph workflow
class ClinicalTrialState(TypedDict):
    messages: Annotated[list, add_messages]
    participant_info: Dict
    eligibility_status: str
    screening_progress: Dict
    next_action: str
    call_metadata: Dict
    conversation_summary: str

class ClinicalTrialAgent:
    """Enhanced LangChain/LangGraph agent for clinical trials with MongoDB integration"""
    
    def __init__(self, openai_api_key: str, db_service: MongoDBService):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o",
            temperature=0.7
        )
        self.db_service = db_service
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        # Initialize checkpointer for state persistence
        self.checkpointer = MemorySaver()
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for clinical trial conversations"""
        workflow = StateGraph(ClinicalTrialState)
        
        # Add nodes for different conversation stages
        workflow.add_node("greeting", self._greeting_handler)
        workflow.add_node("medication_check", self._medication_check)
        workflow.add_node("symptom_assessment", self._symptom_assessment)
        workflow.add_node("general_conversation", self._general_conversation)
        workflow.add_node("call_conclusion", self._call_conclusion)
        
        # Set entry point
        workflow.set_entry_point("greeting")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "greeting",
            self._route_conversation,
            {
                "medication": "medication_check",
                "symptoms": "symptom_assessment", 
                "general": "general_conversation",
                "conclude": "call_conclusion"
            }
        )
        
        workflow.add_conditional_edges(
            "medication_check",
            self._route_conversation,
            {
                "symptoms": "symptom_assessment",
                "general": "general_conversation", 
                "conclude": "call_conclusion"
            }
        )
        
        workflow.add_conditional_edges(
            "symptom_assessment",
            self._route_conversation,
            {
                "medication": "medication_check",
                "general": "general_conversation",
                "conclude": "call_conclusion"
            }
        )
        
        workflow.add_conditional_edges(
            "general_conversation",
            self._route_conversation,
            {
                "medication": "medication_check",
                "symptoms": "symptom_assessment",
                "conclude": "call_conclusion"
            }
        )
        
        workflow.add_edge("call_conclusion", END)
        
        return workflow
    
    def _route_conversation(self, state: ClinicalTrialState) -> str:
        """Route conversation based on content and progress"""
        messages = state["messages"]
        screening_progress = state.get("screening_progress", {})
        
        if not messages:
            return "general"
        
        last_message = messages[-1].content.lower() if messages else ""
        
        # Check for conclusion keywords
        if any(word in last_message for word in ["goodbye", "thank you", "bye", "end", "finish"]):
            return "conclude"
        
        # Check for medication-related keywords
        if any(word in last_message for word in ["medication", "pills", "dose", "medicine", "metformin", "insulin"]):
            return "medication"
        
        # Check for symptom-related keywords  
        if any(word in last_message for word in ["feel", "symptom", "sick", "pain", "tired", "dizzy", "nausea"]):
            return "symptoms"
        
        # Check conversation progress
        if not screening_progress.get("medication_checked"):
            return "medication"
        elif not screening_progress.get("symptoms_checked"):
            return "symptoms"
        else:
            return "general"
    
    async def _greeting_handler(self, state: ClinicalTrialState) -> ClinicalTrialState:
        """Handle initial greeting and setup"""
        participant_name = state.get("participant_info", {}).get("name", "")
        first_name = participant_name.split()[0] if participant_name else "there"
        
        greeting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Lotus, a caring AI assistant conducting a wellness check-in call 
            for a participant in a Type 2 Diabetes clinical trial. 
            
            Keep your greeting warm, professional, and brief. Ask how they're feeling today
            and let them know this is their scheduled check-in call from the study.
            
            Be conversational and empathetic. Keep responses under 100 words."""),
            ("human", f"Greet {first_name} warmly for their diabetes study check-in call.")
        ])
        
        chain = greeting_prompt | self.llm
        response = await chain.ainvoke({})
        
        state["messages"].append(AIMessage(content=response.content))
        state["screening_progress"]["greeted"] = True
        
        return state
    
    async def _medication_check(self, state: ClinicalTrialState) -> ClinicalTrialState:
        """Handle medication compliance questions"""
        messages = state["messages"]
        last_human_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break
        
        medication_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are checking on medication compliance for a diabetes study participant.
            
            Ask about:
            - Whether they took their medication today
            - Any missed doses recently
            - Any difficulties with their medication routine
            - Any side effects they've noticed
            
            Be supportive and non-judgmental. If they missed doses, gently remind them of the 
            importance but don't scold. Keep responses conversational and under 150 words."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = medication_prompt | self.llm
        chat_history = messages[:-1] if len(messages) > 1 else []
        
        response = await chain.ainvoke({
            "input": last_human_message or "Let's talk about your medication routine",
            "chat_history": chat_history
        })
        
        state["messages"].append(AIMessage(content=response.content))
        state["screening_progress"]["medication_checked"] = True
        
        # Save medication compliance info
        await self._save_screening_data(state, "medication_check", {
            "patient_response": last_human_message,
            "ai_response": response.content,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
        return state
    
    async def _symptom_assessment(self, state: ClinicalTrialState) -> ClinicalTrialState:
        """Handle symptom and side effect assessment"""
        messages = state["messages"]
        last_human_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break
        
        symptom_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are assessing symptoms and side effects for a diabetes study participant.
            
            Ask about:
            - How they're feeling overall
            - Any unusual symptoms (fatigue, dizziness, nausea)
            - Blood sugar levels if they monitor them
            - Any concerns about their health
            
            Be empathetic and thorough. If they report concerning symptoms, suggest they 
            contact their study coordinator or doctor. Keep responses under 150 words."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = symptom_prompt | self.llm
        chat_history = messages[:-1] if len(messages) > 1 else []
        
        response = await chain.ainvoke({
            "input": last_human_message or "How have you been feeling lately?",
            "chat_history": chat_history
        })
        
        state["messages"].append(AIMessage(content=response.content))
        state["screening_progress"]["symptoms_checked"] = True
        
        # Save symptom assessment info
        await self._save_screening_data(state, "symptom_assessment", {
            "patient_response": last_human_message,
            "ai_response": response.content,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
        return state
    
    async def _general_conversation(self, state: ClinicalTrialState) -> ClinicalTrialState:
        """Handle general conversation and questions"""
        messages = state["messages"]
        last_human_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break
        
        general_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are having a supportive conversation with a diabetes study participant.
            
            Respond naturally to their questions or concerns. Provide encouragement and support.
            If they have medical questions, remind them to contact their study coordinator or doctor.
            
            Keep the conversation warm and helpful. Be ready to transition to closing the call
            when appropriate. Keep responses under 150 words."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = general_prompt | self.llm
        chat_history = messages[:-1] if len(messages) > 1 else []
        
        response = await chain.ainvoke({
            "input": last_human_message or "Is there anything else you'd like to discuss?",
            "chat_history": chat_history
        })
        
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    async def _call_conclusion(self, state: ClinicalTrialState) -> ClinicalTrialState:
        """Handle call conclusion and summary generation"""
        messages = state["messages"]
        
        conclusion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are concluding a wellness check-in call for a diabetes study participant.
            
            Provide a warm, professional closing:
            - Thank them for their time
            - Remind them when their next check-in will be (typically in 1 week)
            - Encourage them to contact the study team with any questions
            - Wish them well
            
            Keep it brief and positive, under 100 words."""),
            ("human", "Please conclude this check-in call professionally and warmly.")
        ])
        
        chain = conclusion_prompt | self.llm
        response = await chain.ainvoke({})
        
        state["messages"].append(AIMessage(content=response.content))
        
        # Generate conversation summary
        summary = await self._generate_clinical_summary(messages)
        state["conversation_summary"] = summary
        
        # Save final call data
        await self._save_call_completion(state)
        
        return state
    
    async def _generate_clinical_summary(self, messages: List) -> str:
        """Generate clinical summary from conversation"""
        try:
            # Extract conversation content
            conversation_text = " ".join([
                f"{'Patient' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in messages[-20:]  # Last 20 messages
                if isinstance(msg, (HumanMessage, AIMessage))
            ])
            
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are generating a clinical summary for a diabetes study check-in call.
                
                Based on the conversation, create a brief clinical note covering:
                - Medication compliance status
                - Reported symptoms or side effects
                - Patient's overall condition
                - Any concerns or follow-up needed
                - Recommended next steps
                
                Format as a professional clinical note, 3-4 sentences maximum.
                Use clinical terminology where appropriate but keep it clear."""),
                ("human", f"Conversation:{conversation_text}Generate clinical summary:")
            ])
            
            chain = summary_prompt | self.llm
            response = await chain.ainvoke({})
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            return "Call completed - summary generation failed"
    
    async def _save_screening_data(self, state: ClinicalTrialState, screening_type: str, data: Dict):
        """Save screening data to MongoDB"""
        try:
            stream_sid = state.get("call_metadata", {}).get("stream_sid")
            if stream_sid:
                # Update the call record with screening data
                await self.db_service.db.calls.update_one(
                    {"stream_sid": stream_sid},
                    {"$set": {
                        f"screening_data.{screening_type}": data,
                        "last_updated": datetime.datetime.utcnow()
                    }}
                )
        except Exception as e:
            logger.error(f"Error saving screening data: {e}")
    
    async def _save_call_completion(self, state: ClinicalTrialState):
        """Save call completion data to MongoDB"""
        try:
            stream_sid = state.get("call_metadata", {}).get("stream_sid")
            summary = state.get("conversation_summary", "")
            
            if stream_sid:
                # Update call record with completion data
                await self.db_service.db.calls.update_one(
                    {"stream_sid": stream_sid},
                    {"$set": {
                        "call_completed": True,
                        "completion_time": datetime.datetime.utcnow(),
                        "ai_summary": summary,
                        "screening_progress": state.get("screening_progress", {}),
                        "last_updated": datetime.datetime.utcnow()
                    }}
                )
                
                logger.info(f"Saved call completion data for {stream_sid}")
        except Exception as e:
            logger.error(f"Error saving call completion: {e}")
    
    async def process_message(self, user_message: str, session_id: str, participant_info: Dict = None) -> str:
        """Process a user message through the LangGraph workflow"""
        try:
            # Get or create state
            config = {"configurable": {"thread_id": session_id}}
            current_state = await self.app.aget_state(config)
            
            if not current_state.values:
                # Initialize state for new conversation
                initial_state = ClinicalTrialState(
                    messages=[],
                    participant_info=participant_info or {},
                    eligibility_status="unknown",
                    screening_progress={},
                    next_action="greeting",
                    call_metadata={"stream_sid": session_id},
                    conversation_summary=""
                )
            else:
                initial_state = current_state.values
            
            # Add user message if provided
            if user_message:
                initial_state["messages"].append(HumanMessage(content=user_message))
            
            # Run the workflow
            result = await self.app.ainvoke(initial_state, config)
            
            # Return the latest AI message
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            return ai_messages[-1].content if ai_messages else "I'm here to help with your check-in."
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I'm having trouble right now. Let me try to help you anyway."

# Enhanced LLM service that integrates with LangGraph
class EnhancedLangChainLLMService(OpenAILLMService):
    def __init__(self, api_key: str, model: str = "gpt-4o", db_service: MongoDBService = None):
        super().__init__(api_key=api_key, model=model)
        self.clinical_agent = ClinicalTrialAgent(api_key, db_service or MongoDBService())
        self.session_id = None
        self.participant_info = {}
    
    def set_session_info(self, session_id: str, participant_info: Dict = None):
        """Set session ID and participant info"""
        self.session_id = session_id
        self.participant_info = participant_info or {}
    
    async def _process_context(self, context):
        """Override to integrate with LangGraph workflow"""
        try:
            # Get the user's message from context
            messages = context.get_messages()
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if user_messages and self.session_id:
                latest_user_message = user_messages[-1].get("content", "")
                
                # Process through LangGraph
                response = await self.clinical_agent.process_message(
                    latest_user_message, 
                    self.session_id,
                    self.participant_info
                )
                
                # Update context with LangGraph response
                context.add_message({
                    "role": "assistant",
                    "content": response
                })
            
            return await super()._process_context(context)
        except Exception as e:
            logger.error(f"Error in _process_context: {e}")
            # Fallback to standard processing
            return await super()._process_context(context)

# Enhanced audio saving with better MongoDB integration
async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int, stream_sid: str):
    """Enhanced audio saving with participant context"""
    from_number, to_number = call_state.get_phone_numbers(stream_sid)
    
    if len(audio) > 0:
        try:
            # Get participant info from call state for better metadata
            call_data = call_state.get_call_data(stream_sid)
            participant_name = call_data.get("participant_name", "Unknown") if call_data else "Unknown"
            
            metadata = {
                "server_name": server_name,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "participant_name": participant_name,
                "audio_length": len(audio),
                "processing_timestamp": datetime.datetime.utcnow().isoformat()
            }
            
            await db_service.save_call(
                stream_sid=stream_sid, 
                from_number=from_number, 
                to_number=to_number, 
                audio_data=audio, 
                metadata=metadata
            )
            
            logger.info(f"Saved audio to database: {participant_name} ({stream_sid})")
        except Exception as e:
            logger.error(f"Failed to save audio to database: {e}")
    else:
        logger.warning("No audio data to save!")

# Enhanced transcript saving with clinical context
async def save_transcript(stream_sid: str, transcript: str):
    """Enhanced transcript saving with clinical analysis"""
    try:
        from_number, to_number = call_state.get_phone_numbers(stream_sid)
        
        # Save transcript with enhanced metadata
        await db_service.save_call(
            stream_sid=stream_sid, 
            transcript=transcript, 
            from_number=from_number, 
            to_number=to_number
        )
        
        # Add clinical context if available
        call_data = call_state.get_call_data(stream_sid)
        if call_data:
            participant_name = call_data.get("participant_name", "Unknown")
            logger.info(f"Saved transcript for {participant_name} ({stream_sid})")
        else:
            logger.info(f"Saved transcript for stream_sid: {stream_sid}")
            
    except Exception as e:
        logger.error(f"Failed to save transcript to database: {e}")

# Main bot function with enhanced LangChain integration
async def run_enhanced_clinical_bot(websocket: WebSocket, stream_sid: str, testing: bool):
    """Enhanced bot with LangChain/LangGraph integration for clinical trials"""
    
    # Extract participant info from call state
    participant_info = {}
    call_data = call_state.get_call_data(stream_sid)
    if call_data:
        participant_info = {
            "name": call_data.get("participant_name", "Unknown Participant"),
            "phone": call_data.get("to_number", ""),
            "from_phone": call_data.get("from_number", "")
        }
        logger.info(f"Starting clinical bot for {participant_info['name']}")
    
    async def extract_transcript(context):
        """Extract and save transcript with enhanced clinical context"""
        if not context or not hasattr(context, "get_messages"):
            return ""
        
        messages = context.get_messages()
        transcript = ""
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if isinstance(content, list):
                txt_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        txt_content += item.get("text", "")
                content = txt_content
            
            if content and isinstance(content, str):
                # Use more clinical-appropriate labels
                role_label = "PATIENT" if role == "user" else "LOTUS_AI"
                transcript += f"{role_label}: {content}\n\n"
        
        if transcript:
            await save_transcript(stream_sid, transcript)
    
    transport = FastAPIWebsocketTransport(
        websocket, 
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid=stream_sid),
        )
    )

    # Use enhanced LLM service with clinical context
    llm = EnhancedLangChainLLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model="gpt-4o",
        db_service=db_service
    )
    llm.set_session_info(stream_sid, participant_info)
    
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="kdmDKE6EkgrWrrykO9Qt",
        push_silence_after_stopping=False
    )
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"), 
        audio_passthrough=True
    )

    # Clinical trial focused system message
    participant_name = participant_info.get("name", "")
    first_name = participant_name.split()[0] if participant_name and participant_name != "Unknown Participant" else ""
    
    messages = [
        {
            "role": "system",
            "content": f"""You are Lotus, an AI assistant conducting a wellness check-in call for {participant_name or 'a participant'} 
            in a Type 2 Diabetes clinical trial study. This is a routine follow-up to ensure their wellbeing and medication compliance.

            Your objectives:
            1. Conduct a friendly, professional wellness check
            2. Ask about medication compliance and any missed doses
            3. Inquire about symptoms, side effects, or health concerns
            4. Provide supportive, non-judgmental responses
            5. Suggest contacting the study coordinator for any medical concerns
            6. Keep the conversation focused but allow natural flow

            Be warm, empathetic, and professional. Use the participant's first name when appropriate.
            Keep responses conversational and under 150 words unless more detail is needed."""
        }
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        audiobuffer,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await audiobuffer.start_recording()
        
        # Personalized introduction
        intro_message = {
            "role": "system",
            "content": f"Begin the call by greeting {first_name or 'the participant'} warmly and explaining this is their scheduled wellness check-in from the diabetes study."
        }
        messages.append(intro_message)
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # Save transcript and mark call as completed
        await extract_transcript(context)
        
        # Update call state
        call_state.remove_call(stream_sid)
        
        # Mark call as completed in database
        try:
            await db_service.db.calls.update_one(
                {"stream_sid": stream_sid},
                {"$set": {
                    "call_completed": True,
                    "completion_time": datetime.datetime.utcnow(),
                    "call_status": "completed"
                }}
            )
        except Exception as e:
            logger.error(f"Error updating call completion status: {e}")
        
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        logger.debug(f"Audio data received: {len(audio)} bytes for {participant_info.get('name', 'Unknown')}")
        
        # Enhanced server name with participant context
        server_name = f"{participant_info.get('name', 'Unknown').replace(' ', '_')}_{stream_sid}"
        
        try:
            await save_audio(server_name, audio, sample_rate, num_channels, stream_sid)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
    
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)