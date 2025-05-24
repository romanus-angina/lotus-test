"use client";
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import NavbarSimple from '../../components/NavbarSimple';
import PatientTableSimple from '../../components/PatientTableSimple';
import AddPatientModalSimple from '../../components/AddPatientModalSimple';
import '../../styles/MinimalDashboard.css';

const DashboardPage = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const router = useRouter();

  // IMPORTANT: Make sure this points to your Lotus server
  const API_BASE_URL = 'https://lotustest.ngrok.io';
  
  console.log("🌐 Dashboard using API_BASE_URL:", API_BASE_URL);

  // Test server connection
  const testServerConnection = async () => {
    try {
      console.log("🔍 Testing server connection to:", API_BASE_URL);
      const response = await fetch(`${API_BASE_URL}/api/server-identity`);
      const data = await response.json();
      console.log("🏥 Server identity response:", data);
      
      if (data.server !== "LOTUS_LOCAL_SERVER") {
        console.error("❌ Connected to wrong server!", data);
        setError(`Connected to wrong server: ${data.server}. Expected LOTUS_LOCAL_SERVER.`);
      } else {
        console.log("✅ Successfully connected to LOTUS server");
      }
    } catch (err) {
      console.error("❌ Server connection test failed:", err);
      setError(`Cannot connect to server: ${err.message}. Is Lotus server running?`);
    }
  };

const fetchPatientData = async () => {
  setIsLoading(true);
  setError('');
  try {
    console.log("Fetching patient data from:", `${API_BASE_URL}/api/calls`);
    
    const response = await fetch(`${API_BASE_URL}/api/calls`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Failed to fetch data: ${response.status}`);
    }
    const data = await response.json();

    console.log("Raw API response:", data);

    // Process calls with timing awareness
    const patientMap = new Map();
    
    if (data.calls && Array.isArray(data.calls)) {
      data.calls.forEach((call, index) => {
        const phoneNumber = call.to_number;
        if (!phoneNumber) {
          console.log(`Skipping call ${index + 1} - no phone number`);
          return;
        }

        const patientName = call.contact_name || call.participant_name || 'Unknown Patient';
        const callDate = call.created_at ? new Date(call.created_at) : null;
        const completionTime = call.completion_time ? new Date(call.completion_time) : null;
        const hasTranscript = !!(call.transcript && call.transcript.trim());
        
        // TIMING-AWARE: Determine summary status based on call lifecycle
        let summary = "No summary yet.";
        let summaryStatus = "none";
        
        if (call.clinical_summary && call.clinical_summary.trim() !== "") {
          summary = call.clinical_summary;
          summaryStatus = "completed";
        } else if (call.ai_summary && call.ai_summary.trim() !== "") {
          summary = call.ai_summary;
          summaryStatus = "completed";
        } else if (call.call_completed && hasTranscript) {
          // Call completed with transcript but no summary yet - summary is being generated
          summary = "Summary being generated... (refresh in a moment)";
          summaryStatus = "generating";
        } else if (hasTranscript) {
          // Has transcript but call might not be marked complete
          summary = "Call completed - processing summary...";
          summaryStatus = "processing";
        } else if (call.call_completed) {
          // Call completed but no transcript
          summary = "Call completed - no transcript available";
          summaryStatus = "no_transcript";
        }

        console.log(`Call ${index + 1}: ${patientName} - Status: ${summaryStatus}, Summary: "${summary.substring(0, 50)}..."`);

        // Keep the call with the best status (completed > generating > processing > none)
        const statusPriority = {
          "completed": 4,
          "generating": 3,
          "processing": 2,
          "no_transcript": 1,
          "none": 0
        };

        const existingPatient = patientMap.get(phoneNumber);
        
        if (!existingPatient) {
          patientMap.set(phoneNumber, {
            id: call._id || call.stream_sid,
            name: patientName,
            phoneNumber: phoneNumber,
            lastCallDate: callDate,
            callSummary: summary,
            summaryStatus: summaryStatus,
            transcript: call.transcript || "",
            hasTranscript: hasTranscript,
            rawCallData: call,
          });
        } else {
          const existingPriority = statusPriority[existingPatient.summaryStatus] || 0;
          const currentPriority = statusPriority[summaryStatus] || 0;
          
          const shouldReplace = (
            currentPriority > existingPriority ||
            (currentPriority === existingPriority && callDate && callDate > (existingPatient.lastCallDate || 0))
          );
          
          if (shouldReplace) {
            patientMap.set(phoneNumber, {
              id: call._id || call.stream_sid,
              name: patientName,
              phoneNumber: phoneNumber,
              lastCallDate: callDate,
              callSummary: summary,
              summaryStatus: summaryStatus,
              transcript: call.transcript || "",
              hasTranscript: hasTranscript,
              rawCallData: call,
            });
            console.log(`UPDATED patient: ${patientName} with status: ${summaryStatus}`);
          }
        }
      });
    }
    
    const processedPatients = Array.from(patientMap.values()).sort((a,b) => (b.lastCallDate || 0) - (a.lastCallDate || 0));
    
    console.log("FINAL RESULT:", processedPatients.map(p => ({
      name: p.name,
      summary: p.callSummary.substring(0, 50) + "...",
      status: p.summaryStatus,
      hasTranscript: p.hasTranscript
    })));
    
    setPatients(processedPatients);

  } catch (err) {
    console.error("Error fetching patient data:", err);
    setError(err.message);
    setPatients([]);
  } finally {
    setIsLoading(false);
  }
};

  useEffect(() => {
    const authStatus = sessionStorage.getItem('isStaffAuthenticated');
    if (authStatus === 'true') {
      setIsAuthenticated(true);
      testServerConnection(); // Test connection first
      fetchPatientData();
    } else {
      router.push('/login');
    }
  }, [router]);

  const handleAddPatientSuccess = () => {
    setShowAddPatientModal(false);
    fetchPatientData();
  };

  if (!isAuthenticated) {
    return <div className="loading-state">Redirecting to login...</div>;
  }

  return (
    <>
      <NavbarSimple />
      <main className="dashboard-main">
      <div className="dashboard-header">
      <h1>Patient Dashboard</h1>
      <div style={{ display: 'flex', gap: '10px' }}>
        <button 
          onClick={fetchPatientData} 
          className="add-patient-btn"
          style={{ backgroundColor: '#009688' }}
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing...' : '🔄 Refresh'}
        </button>
        <button onClick={() => setShowAddPatientModal(true)} className="add-patient-btn">
          + Add Patient
        </button>
      </div>
    </div>

        {isLoading && <p className="loading-state">Loading patient data...</p>}
        {error && <p className="status-message-global error">Error: {error}</p>}
        
        {!isLoading && !error && (
          <PatientTableSimple
            patients={patients}
            apiBaseUrl={API_BASE_URL}
            onRefreshNeeded={fetchPatientData}
          />
        )}
      </main>
      {showAddPatientModal && (
        <AddPatientModalSimple
          onClose={() => setShowAddPatientModal(false)}
          onAddSuccess={handleAddPatientSuccess}
          apiBaseUrl={API_BASE_URL}
        />
      )}
    </>
  );
};

export default DashboardPage;