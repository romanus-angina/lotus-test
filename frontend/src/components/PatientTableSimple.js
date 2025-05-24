"use client";
import React, { useState } from 'react';
import TranscriptModal from './TranscriptModal';
import '../styles/MinimalDashboard.css';

const PatientTableSimple = ({ patients, apiBaseUrl, onRefreshNeeded }) => {
  const [callingPatientPhone, setCallingPatientPhone] = useState(null);
  const [callError, setCallError] = useState('');
  const [callSuccess, setCallSuccess] = useState('');
  const [selectedPatient, setSelectedPatient] = useState(null);

  const handleTriggerCall = async (patient) => {
    if (!patient.phoneNumber) {
      setCallError(`Phone number missing for ${patient.name}.`);
      return;
    }
    
    console.log("Starting call process for:", patient.name);
    setCallingPatientPhone(patient.phoneNumber);
    setCallError('');
    setCallSuccess('');

    try {
      // Server identity check
      const serverCheckUrl = `${apiBaseUrl}/api/server-identity`;
      const serverResponse = await fetch(serverCheckUrl);
      const serverData = await serverResponse.json();
      
      if (serverData.server !== "LOTUS_LOCAL_SERVER") {
        setCallError(`Wrong server detected: ${serverData.server}`);
        return;
      }
      
      // Make the call
      const response = await fetch(`${apiBaseUrl}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to: patient.phoneNumber,
          participant_name: patient.name
        }),
      });

      const data = await response.json();
      
      if (!response.ok || !data.sid) {
        throw new Error(data.message || 'Failed to initiate call.');
      }
      
      setCallSuccess(`Call initiated to ${patient.name}. Summary will update after call completion.`);
      
      // Auto-refresh after call to show updated summaries
      setTimeout(() => {
        onRefreshNeeded();
        setCallSuccess('');
      }, 30000);

    } catch (error) {
      console.error('Error triggering call:', error);
      setCallError(`Error calling ${patient.name}: ${error.message}`);
    } finally {
      setCallingPatientPhone(null);
    }
  };

  const handleViewTranscript = (patient) => {
    setSelectedPatient(patient);
  };
  
  // Clear messages after a few seconds
  React.useEffect(() => {
    if (callError || callSuccess) {
      const timer = setTimeout(() => {
        setCallError('');
        setCallSuccess('');
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [callError, callSuccess]);

  if (!patients || patients.length === 0) {
    return <p className="no-data-state">No patient records found. Add a patient to get started.</p>;
  }

  return (
    <>
      {callError && <p className="status-message-global error">{callError}</p>}
      {callSuccess && <p className="status-message-global success">{callSuccess}</p>}
      
      <div className="patient-table-wrapper">
        <table className="patient-table">
          <thead>
            <tr>
              <th>Patient Name & Phone</th>
              <th>Date of Last Call</th>
              <th>Call Summary</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((patient) => {
              // Visual indicator for summary status
              const getSummaryStyle = () => {
                if (patient.summaryStatus === 'completed') return { backgroundColor: '#e8f5e8' };
                if (patient.summaryStatus === 'generating') return { backgroundColor: '#fff3cd' };
                if (patient.summaryStatus === 'processing') return { backgroundColor: '#d1ecf1' };
                return { backgroundColor: '#f8f9fa' };
              };

              return (
                <tr key={patient.id || patient.phoneNumber}>
                  <td>
                    <div>{patient.name}</div>
                    <div style={{ fontSize: '0.8rem', color: '#546e7a' }}>{patient.phoneNumber}</div>
                  </td>
                  <td>{patient.lastCallDate ? new Date(patient.lastCallDate).toLocaleDateString() : 'N/A'}</td>
                  <td className="call-summary-cell" title={patient.callSummary || 'No summary available.'}>
                    <div style={getSummaryStyle()}>
                      {patient.callSummary || 'No summary available.'}
                    </div>
                  </td>
                  <td className="actions-cell">
                    <button
                      onClick={() => handleTriggerCall(patient)}
                      disabled={callingPatientPhone === patient.phoneNumber}
                    >
                      {callingPatientPhone === patient.phoneNumber ? 'Calling...' : 'Trigger Call'}
                    </button>
                    
                    <button
                      onClick={() => handleViewTranscript(patient)}
                      className="view-transcript-btn"
                      disabled={!patient.hasTranscript}
                      title={patient.hasTranscript ? 'View call transcript' : 'No transcript available'}
                    >
                      View Transcript
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Transcript Modal */}
      {selectedPatient && (
        <TranscriptModal
          patient={selectedPatient}
          onClose={() => setSelectedPatient(null)}
        />
      )}
    </>
  );
};

export default PatientTableSimple;