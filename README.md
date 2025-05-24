# Lotus: AI-Powered Clinical Trial Platform
This project implements a comprehensive platform for managing clinical trials, leveraging AI for automated patient check-in calls and data analysis.  The system automates patient interaction, generates clinical summaries, and provides tools for managing patient data and analyzing call transcripts.

## Features
* **Automated Patient Check-in Calls:** Uses Twilio to initiate outbound calls and a custom AI chatbot to conduct wellness check-ins.
* **AI-Powered Conversational Bot:**  A sophisticated chatbot, powered by OpenAI's GPT models, engages patients in a conversational manner, collecting relevant clinical data.
* **Speech-to-Text (STT) and Text-to-Speech (TTS):** Integrates Deepgram for STT and ElevenLabs for TTS, enabling seamless audio processing.
* **Clinical Summary Generation:** Automatically generates concise clinical summaries from call transcripts using OpenAI's GPT models.
* **Secure Data Storage:** Stores call recordings, transcripts, and metadata in a MongoDB database.
* **Web Interface:**  Provides a user-friendly web interface for viewing patient data, call summaries, and transcripts.
* **Background Analysis:** Enables background processing for large-scale clinical data analysis.
* **Contact Management:** Adds and manages clinical trial participant information.
* **Call Statistics:** Tracks key metrics such as total calls, compliant participants, and compliance rates.

## Usage
The platform consists of a backend (Python/FastAPI) and a frontend (Next.js/React).  The backend handles call initiation, AI processing, and data storage.  The frontend provides a user interface for managing patients and viewing call data.

To initiate a call:

1.  **POST** to `/call` with a JSON payload containing the `to` (patient phone number) and optionally `from` (Twilio number) and `participant_name`.  

```json
{
  "to": "+15551234567",
  "participant_name": "John Doe"
}
```

The backend will use the Twilio API to make the call. The call will connect to the websocket endpoint `/ws` for real-time audio processing.

## Installation
**Backend:**

1.  Clone the repository: `git clone <repository_url>`
2.  Navigate to the `backend` directory.
3.  Create a `.env` file with the following environment variables:

```
OPENAI_API_KEY=<your_openai_api_key>
ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
DEEPGRAM_API_KEY=<your_deepgram_api_key>
TWILIO_ACCOUNT_SID=<your_twilio_account_sid>
TWILIO_AUTH_TOKEN=<your_twilio_auth_token>
TWILIO_PHONE_NUMBER=<your_twilio_phone_number>
MONGODB_URI=<your_mongodb_connection_string>
WEBHOOK_BASE_URL=<your_webhook_base_url>
```

4.  Install dependencies: `pip install -r requirements.txt`
5. Run:  `ngrok http 8765 --subdomain <your_subdomain>`
Note: <your_webhook_base_url> will be your ngrok url
6. In a new, terminal Run the server: `python server.py`

**Frontend:**

1.  Navigate to the `frontend` directory.
2.  Install dependencies: `npm install`
3.  Run the development server: `npm run dev`

## Technologies Used
* **Python:** The backend is written in Python, leveraging its extensive libraries for AI, web development, and data processing.
* **FastAPI:**  A modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.
* **Next.js:** React framework for building user interfaces.
* **React:**  A JavaScript library for building user interfaces.
* **Twilio:** A cloud communications platform for initiating and managing phone calls.
* **OpenAI:** Provides access to powerful language models (GPT-4o) for conversational AI and clinical summary generation.
* **Deepgram:**  A speech-to-text API for transcribing call audio.
* **ElevenLabs:**  A text-to-speech API for generating AI voice responses.
* **MongoDB:** A NoSQL database for storing call data.
* **Pipecat:** A Python library for building audio pipelines.
* **Loguru:** Python logging library.
* **Motor:** Asynchronous Python driver for MongoDB.
* **BSON:** Binary JSON encoding for MongoDB.
* **aiofiles:** Asynchronous file I/O support.

## Configuration
The backend uses a `.env` file for configuration.  See the "Installation" section for details on required environment variables.

## API Documentation
The backend exposes several API endpoints:

* `/call`: Initiates a call (POST).
* `/api/calls`: Retrieves a list of calls (GET).
* `/api/calls/{stream_sid}/audio`: Retrieves the audio for a specific call (GET).
* `/api/calls/{phone_number}/notes`: Saves clinical notes (POST).
* `/api/server-identity`: Returns server identification.
* `/api/calls/{stream_sid}/analyze`: Analyzes call transcript for clinical eligibility (POST).
* `/api/analyze-all`: Initiates background analysis of all call transcripts (POST).
* `/api/contacts`: Adds a new contact (POST).
* `/api/call-stats`: Retrieves clinical trial call statistics (GET).
* `/api/health`: Health check endpoint (GET)

## Dependencies
The project dependencies are listed in `backend/requirements.txt` and `frontend/package.json`.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Testing
Unit and integration tests are recommended but not fully implemented in this codebase.  Future improvements should include more comprehensive tests.

*README.md was made with [Etchr](https://etchr.dev)*