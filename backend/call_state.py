CALL_DATA = {}

def store_call(call_sid, from_number, to_number):
    """Store a new call's phone numbers"""
    CALL_DATA[call_sid] = {
        "from_number": from_number,
        "to_number": to_number
    }
    print(f"Stored call {call_sid}: from={from_number}, to={to_number}", flush=True)

def link_stream_sid(call_sid, stream_sid):
    """Create a mapping from stream_sid to the same call data"""
    if call_sid in CALL_DATA:
        # Copy the same data under the stream_sid key
        CALL_DATA[stream_sid] = CALL_DATA[call_sid]
        print(f"Linked stream_sid {stream_sid} to call_sid {call_sid}", flush=True)

def get_call_data(sid):
    """Get call data by either call_sid or stream_sid"""
    return CALL_DATA.get(sid, {})

def get_phone_numbers(sid):
    """Get phone numbers for a specific call/stream SID"""
    data = CALL_DATA.get(sid, {})
    return data.get("from_number"), data.get("to_number")

def remove_call(sid):
    """Clean up call data when finished"""
    if sid in CALL_DATA:
        call_data = CALL_DATA.pop(sid)
        
        # If this is a call_sid, also clean up any linked stream_sid
        # (and vice versa)
        for other_sid in list(CALL_DATA.keys()):
            if CALL_DATA[other_sid] is call_data:  # Same object reference
                CALL_DATA.pop(other_sid)
        
        print(f"Removed call data for SID: {sid}", flush=True)