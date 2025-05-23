import os
import io
import datetime
from typing import Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson.binary import Binary
from loguru import logger

class MongoDBService:
    def __init__(self):
        connection_string = os.getenv("MONGODB_URI")
        if not connection_string:
            logger.error("MONGODB_URI environment variable is not set.")
            raise ValueError("MONGODB_URI environment variable is not set.")
        
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client.lotustest_db
        self.calls = self.db.calls

        logger.info("MongoDB connection established.")

    async def save_call(self, stream_sid: str, from_number: Optional[str], to_number: Optional[str], audio_data: Optional[bytes] = None, transcript: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save a call record to the database.

        Inputs:
        - stream_sid: Twilio stream SID for each individual calll
        - audio_data: Optional raw audio bytes
        - transcript: Optional call transcript text
        - metadata: Additional call metadata e.g. call number 
        
        Returns:
        - The MongoDB document ID of the saved call
        """
        try:
            if metadata is None:
                metadata = {}

            timestamp = datetime.datetime.utcnow()
            existing_record = await self.calls.find_one({"stream_sid": stream_sid})

            if existing_record:
                update_data = {
                    "last_updated": timestamp,
                    **({
                        "audio_data": Binary(audio_data),
                        "audio_size": len(audio_data),
                        } if audio_data else {}),
                        **({
                        "from_number": from_number,
                        } if from_number else {}),
                        **({
                        "to_number": to_number,
                        } if to_number else {}),
                        **({
                        "transcript": transcript,
                        } if transcript else {}),
                        }
                
                if metadata:
                    for key, value in metadata.items():
                        update_data[f"metadata.{key}"] = value

                await self.calls.update_one({"_id": existing_record["_id"]}, {"$set": update_data})
                logger.info(f"Updated existing call record for stream_sid: {stream_sid}")
                return str(existing_record["_id"])
            else:
                record = {
                    "stream_sid": stream_sid,
                    "from_number": from_number if from_number else None,
                    "to_number": to_number if to_number else None,
                    "audio_data": Binary(audio_data) if audio_data else None,
                    "audio_size": len(audio_data) if audio_data else 0,
                    "transcript": transcript if transcript else None,
                    "metadata": metadata if metadata else {},
                    "created_at": timestamp,
                    "last_updated": timestamp
                }
                result = await self.calls.insert_one(record)
                logger.info(f"Inserted new call record for stream_sid: {stream_sid}")
                return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving call record: {e}")
            return None
        
    async def get_call(self, stream_sid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a call record from the database.

        Inputs:
        - stream_sid: Twilio stream SID for each individual call

        Returns:
        - The call record as a dictionary, or None if not found
        """
        try:
            record = await self.calls.find_one({"stream_sid": stream_sid})
            if record:
                record["_id"] = str(record["_id"])  # Convert ObjectId to string
                logger.info(f"Retrieved call record for stream_sid: {stream_sid}")
                return record
            else:
                logger.warning(f"No call record found for stream_sid: {stream_sid}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving call record: {e}")
            return None