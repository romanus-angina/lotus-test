import argparse
import sys
import json
import uvicorn
import datetime
from fastapi import FastAPI, HTTPException, WebSocket, Request, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from fastapi.responses import StreamingResponse

# Import your existing modules
from bot import run_enhanced_clinical_bot
from caller import make_call
from db_service import MongoDBService
import call_state
import os
import asyncio
import io
import wave
from openai import OpenAI

# Initialize services
db_service = MongoDBService()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Clinical Trial Platform",
    description="Enhanced clinical trial management with LangChain/LangGraph AI integration",
    version="2.0.0"
)
app.state.testing = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml_response(request: Request):
    """Return TwiML instructions when Twilio connects the call"""
    print(f"Received {request.method} request to /twiml", flush=True)
    try:
        content = open("templates/streams.xml").read()
        print(f"TwiML content: {content}", flush=True)
        return HTMLResponse(
            content=content,
            media_type="application/xml"
        )
    except Exception as e:
        print(f"Error serving TwiML: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with LangChain integration"""
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    
    print("Call data:", call_data, flush=True)
    
    stream_sid = call_data.get("start", {}).get("streamSid")
    call_sid = call_data.get("start", {}).get("callSid")
    
    print(f"Stream SID: {stream_sid}", flush=True)
    print(f"Call SID: {call_sid}", flush=True)
    
    # Link stream to call in state management
    if call_sid:
        call_state.link_stream_sid(call_sid, stream_sid)
        print(f"Linked stream {stream_sid} to call {call_sid}", flush=True)
    
    # Update call status in MongoDB
    try:
        await db_service.db.calls.update_one(
            {"stream_sid": stream_sid},
            {"$set": {
                "call_sid": call_sid,
                "call_status": "in_progress",
                "call_started": datetime.datetime.utcnow()
            }},
            upsert=True
        )
    except Exception as e:
        print(f"Error updating call status: {e}", flush=True)
    
    # Run the enhanced clinical bot with LangChain
    await run_enhanced_clinical_bot(websocket, stream_sid, app.state.testing)

@app.post("/call")
async def api_call(data: dict = Body(...)):
    """Enhanced endpoint to initiate a call with participant tracking"""
    try:
        to_number = data.get("to")
        from_number = data.get("from") or os.getenv("TWILIO_PHONE_NUMBER")
        participant_name = data.get("participant_name", "Unknown Participant")

        if not to_number:
            raise HTTPException(status_code=400, detail="Phone number is required")
        
        if not from_number:
            raise HTTPException(status_code=500, detail="TWILIO_PHONE_NUMBER missing")

        # Validate phone number format
        if not to_number.startswith("+"):
            to_number = "+" + to_number
        
        print(f"Attempting to call {participant_name} at {to_number} from {from_number}", flush=True)
        
        # Make the call using your existing caller
        sid = make_call(
            to_number=to_number,
            from_number=from_number,
            webhook_url= "https://lotustest.ngrok.io/twiml",
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN")
        )

        if sid:
            # Enhanced call data storage with participant info
            call_state.CALL_DATA[sid]["participant_name"] = participant_name
            
            # Save initial call record to MongoDB with participant context
            try:
                await db_service.save_call(
                    stream_sid=f"pending_{sid}",  # Temporary until we get real stream_sid
                    from_number=from_number,
                    to_number=to_number,
                    metadata={
                        "participant_name": participant_name,
                        "call_sid": sid,
                        "call_initiated": datetime.datetime.utcnow().isoformat(),
                        "call_type": "clinical_trial_checkin"
                    }
                )
            except Exception as e:
                print(f"Error saving initial call record: {e}", flush=True)
            
            return {
                "message": f"Call initiated successfully to {participant_name}",
                "sid": sid,
                "participant_name": participant_name,
                "status": "success"
            }
        else:
            return {"message": "Failed to get call SID", "status": "error"}
            
    except Exception as e:
        print(f"Error in api_call: {str(e)}", flush=True)
        return {"message": f"Error: {str(e)}", "status": "error"}

# Enhanced API endpoints for clinical dashboard integration
@app.get("/api/calls")
async def get_calls():
    """Get all calls with enhanced clinical context"""
    try:
        # Get calls with clinical screening data
        calls = await db_service.db.calls.find(
            {},
            {"audio_data": 0}  # Exclude large audio data
        ).sort("created_at", -1).to_list(length=100)
        
        # Enhance call data with clinical context
        enhanced_calls = []
        for call in calls:
            call["_id"] = str(call["_id"])
            
            # Add clinical summary if available
            if call.get("ai_summary"):
                call["clinical_summary"] = call["ai_summary"]
            elif call.get("transcript"):
                # Generate quick summary if not exists
                call["clinical_summary"] = await generate_quick_summary(call["transcript"])
            
            # Add call duration
            if call.get("call_started") and call.get("completion_time"):
                duration = call["completion_time"] - call["call_started"]
                call["call_duration"] = str(duration)
            
            # Add screening progress
            screening = call.get("screening_data", {})
            call["medication_checked"] = bool(screening.get("medication_check"))
            call["symptoms_checked"] = bool(screening.get("symptom_assessment"))
            
            enhanced_calls.append(call)
        
        # Get contacts without calls (for dashboard)
        existing_numbers = {call.get("to_number") for call in enhanced_calls if call.get("to_number")}
        contacts_without_calls = await db_service.db.contacts.find(
            {"phone_number": {"$nin": list(existing_numbers)}}
        ).to_list(length=100)
        
        # Create virtual call entries for contacts without calls
        for contact in contacts_without_calls:
            virtual_call = {
                "_id": str(contact["_id"]),
                "stream_sid": f"contact_{str(contact['_id'])}",
                "to_number": contact["phone_number"],
                "contact_name": contact["name"],
                "participant_name": contact["name"],
                "created_at": contact.get("created_at", datetime.datetime.utcnow()).isoformat(),
                "virtual_contact": True,
                "call_status": "not_called",
                "clinical_summary": "No calls yet - ready for initial contact"
            }
            enhanced_calls.append(virtual_call)
        
        return {"calls": enhanced_calls}
    except Exception as e:
        print(f"Error fetching calls: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/calls/{stream_sid}/audio")
async def get_call_audio(stream_sid: str):
    """Stream audio recording for a specific call"""
    try:
        call = await db_service.get_call(stream_sid)
        if not call or "audio_data" not in call or not call["audio_data"]:
            raise HTTPException(status_code=404, detail=f"Audio not found for call: {stream_sid}")
        
        # Create audio stream with WAV header
        audio_data = call["audio_data"]
        has_wav_header = len(audio_data) > 4 and audio_data[:4] == b'RIFF'
        
        if not has_wav_header:
            # Add WAV header for better compatibility
            sample_rate = call.get("metadata", {}).get("sample_rate", 8000)
            num_channels = call.get("metadata", {}).get("num_channels", 1)
            
            wav_stream = io.BytesIO()
            with wave.open(wav_stream, 'wb') as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            wav_stream.seek(0)
            audio_stream = wav_stream
        else:
            audio_stream = io.BytesIO(audio_data)
        
        # Get participant name for filename
        participant_name = call.get("metadata", {}).get("participant_name", "Unknown")
        filename = f"{participant_name.replace(' ', '_')}_{stream_sid}.wav"
        
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(audio_stream.getbuffer().nbytes),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error streaming audio: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/calls/{phone_number}/notes")
async def save_call_notes(phone_number: str, data: dict = Body(...)):
    """Save clinical notes for calls to a specific phone number"""
    try:
        notes = data.get("notes", "")
        clinical_notes = data.get("clinical_notes", "")
        
        update_data = {"notes": notes}
        if clinical_notes:
            update_data["clinical_notes"] = clinical_notes
            update_data["notes_updated"] = datetime.datetime.utcnow()
        
        result = await db_service.db.calls.update_many(
            {"to_number": phone_number},
            {"$set": update_data}
        )
        
        return {
            "success": True, 
            "message": "Clinical notes saved successfully", 
            "updated_count": result.modified_count
        }
    except Exception as e:
        print(f"Error saving clinical notes: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Enhanced eligibility analysis with clinical context
@app.post("/api/calls/{stream_sid}/analyze")
async def analyze_transcript(stream_sid: str, data: dict = Body(...)):
    """Analyze transcript with clinical trial context"""
    try:
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Analysis prompt is required")
        
        call = await db_service.get_call(stream_sid)
        if not call:
            raise HTTPException(status_code=404, detail=f"Call not found: {stream_sid}")
        
        if not call.get("transcript"):
            raise HTTPException(status_code=400, detail="No transcript available for analysis")
        
        # Enhanced clinical analysis prompt
        clinical_system_message = (
            "You are a clinical research AI analyzing patient check-in calls for a Type 2 Diabetes study. "
            "Based on the transcript and specific prompt, provide a clinical assessment. "
            "Consider medication compliance, reported symptoms, patient wellbeing, and any red flags. "
            "Respond with 'Yes' or 'No' followed by a brief clinical rationale in parentheses."
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": clinical_system_message},
                {"role": "user", "content": f"Analysis prompt: {prompt}\n\nPatient call transcript:\n{call['transcript']}"}
            ],
            max_tokens=100,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Extract Yes/No and rationale
        binary_result = "Yes" if result.lower().startswith("yes") else "No"
        
        # Save analysis with clinical context
        await db_service.db.calls.update_one(
            {"stream_sid": stream_sid},
            {"$set": {
                "clinical_analysis": {
                    "prompt": prompt,
                    "result": binary_result,
                    "full_response": result,
                    "timestamp": datetime.datetime.utcnow(),
                    "analysis_type": "clinical_eligibility"
                }
            }}
        )
        
        return {"success": True, "result": binary_result, "clinical_notes": result}
    except Exception as e:
        print(f"Error analyzing transcript: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/analyze-all")
async def analyze_all_calls(background_tasks: BackgroundTasks, data: dict = Body(...)):
    """Analyze all clinical trial calls in background"""
    try:
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Analysis prompt is required")
        
        background_tasks.add_task(analyze_all_clinical_transcripts, prompt)
        
        return {"success": True, "message": "Clinical analysis started in background"}
    except Exception as e:
        print(f"Error starting analysis: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def analyze_all_clinical_transcripts(prompt: str):
    """Background task for clinical analysis"""
    try:
        print(f"Starting clinical analysis with prompt: {prompt}", flush=True)
        
        # Get most recent call per participant
        pipeline = [
            {"$match": {"transcript": {"$exists": True, "$ne": ""}}},
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$to_number",
                "latest_call": {"$first": "$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest_call"}}
        ]
        
        calls = await db_service.db.calls.aggregate(pipeline).to_list(length=1000)
        
        print(f"Analyzing {len(calls)} clinical calls...", flush=True)
        
        for call in calls:
            try:
                stream_sid = call["stream_sid"]
                transcript = call["transcript"]
                participant_name = call.get("metadata", {}).get("participant_name", "Unknown")
                
                print(f"Analyzing call for {participant_name} ({stream_sid})", flush=True)
                
                # Clinical analysis
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are analyzing clinical trial check-in calls. Assess based on the prompt and provide Yes/No with brief clinical reasoning."},
                        {"role": "user", "content": f"Clinical assessment prompt: {prompt}\n\nParticipant: {participant_name}\nTranscript:\n{transcript}"}
                    ],
                    max_tokens=150,
                    temperature=0
                )
                
                result = response.choices[0].message.content.strip()
                binary_result = "Yes" if result.lower().startswith("yes") else "No"
                
                # Save clinical analysis
                await db_service.db.calls.update_one(
                    {"stream_sid": stream_sid},
                    {"$set": {
                        "clinical_analysis": {
                            "prompt": prompt,
                            "result": binary_result,
                            "clinical_assessment": result,
                            "timestamp": datetime.datetime.utcnow(),
                            "participant_name": participant_name
                        }
                    }}
                )
                
                print(f"Clinical analysis for {participant_name}: {binary_result}", flush=True)
            except Exception as e:
                print(f"Error analyzing call {call.get('stream_sid')}: {str(e)}", flush=True)
    except Exception as e:
        print(f"Error in clinical analysis task: {str(e)}", flush=True)

@app.post("/api/contacts")
async def add_contact(data: dict = Body(...)):
    """Add new clinical trial participant"""
    try:
        phone_number = data.get("phone_number")
        name = data.get("name", "Unnamed Participant")
        study_id = data.get("study_id", "diabetes_study_001")
        
        if not phone_number:
            raise HTTPException(status_code=400, detail="Phone number is required")
        
        # Create participant record
        result = await db_service.db.contacts.insert_one({
            "phone_number": phone_number,
            "name": name,
            "study_id": study_id,
            "participant_status": "enrolled",
            "created_at": datetime.datetime.utcnow(),
            "last_contact": None
        })
        
        # Update existing calls with participant name
        await db_service.db.calls.update_many(
            {"to_number": phone_number},
            {"$set": {
                "contact_name": name,
                "participant_name": name,
                "study_id": study_id
            }}
        )
        
        return {
            "success": True, 
            "message": f"Participant {name} added successfully", 
            "id": str(result.inserted_id)
        }
    except Exception as e:
        print(f"Error adding participant: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/call-stats")
async def get_clinical_call_stats():
    """Get clinical trial call statistics"""
    try:
        start_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)
        
        calls = await db_service.db.calls.find({"created_at": {"$gte": start_date}}).to_list(length=1000)
        
        daily_stats = {}
        total_duration = 0
        compliant_count = 0
        
        for call in calls:
            call_date = call.get("created_at")
            if not call_date:
                continue
                
            date_str = call_date.strftime("%Y-%m-%d")
            
            if date_str not in daily_stats:
                daily_stats[date_str] = {"totalCalls": 0, "compliantParticipants": 0}
            
            daily_stats[date_str]["totalCalls"] += 1
            
            # Check for medication compliance indicators
            transcript = call.get("transcript", "").lower()
            if any(phrase in transcript for phrase in ["taking medication", "took my pills", "compliant", "on schedule"]):
                daily_stats[date_str]["compliantParticipants"] += 1
                compliant_count += 1
        
        # Build results for last 7 days
        results = []
        current_date = datetime.datetime.utcnow()
        
        for i in range(7):
            date = current_date - datetime.timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            if date_str in daily_stats:
                results.insert(0, {
                    "date": date_str,
                    "totalCalls": daily_stats[date_str]["totalCalls"],
                    "compliantParticipants": daily_stats[date_str]["compliantParticipants"]
                })
            else:
                results.insert(0, {
                    "date": date_str,
                    "totalCalls": 0,
                    "compliantParticipants": 0
                })
        
        total_calls = len(calls)
        compliance_rate = (compliant_count / total_calls) * 100 if total_calls > 0 else 0
        
        return {
            "data": results,
            "complianceRate": compliance_rate,
            "totalParticipants": total_calls
        }
    except Exception as e:
        print(f"Error getting clinical stats: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Utility function for quick summary generation
async def generate_quick_summary(transcript: str) -> str:
    """Generate quick clinical summary if not exists"""
    try:
        if len(transcript) < 50:
            return "Brief call - insufficient content for summary"
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate a brief clinical summary (1-2 sentences) of this diabetes study check-in call transcript."},
                {"role": "user", "content": transcript[-1000:]}  # Last 1000 chars
            ],
            max_tokens=100,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Call completed - summary generation failed: {str(e)[:50]}"

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check with system status"""
    try:
        # Test database connection
        db_test = await db_service.db.calls.count_documents({})
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow(),
            "version": "2.0.0",
            "features": ["langchain", "langgraph", "clinical_trials", "mongodb"],
            "database_calls": db_test,
            "services": {
                "mongodb": "connected",
                "openai": "available",
                "langchain": "active"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow()
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Clinical Trial Server")
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Run the server in testing mode",
        default=False
    )
    args, _ = parser.parse_known_args()
    app.state.testing = args.testing
    
    print("🚀 Starting Enhanced Clinical Trial Server", flush=True)
    print("🧠 Features: LangChain + LangGraph + MongoDB + Clinical AI", flush=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8765)