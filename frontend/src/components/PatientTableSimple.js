"use client";
import React, { useState } from 'react';
import '../styles/MinimalDashboard.css'; // Using styles from here

const PatientTableSimple = ({ patients, apiBaseUrl, onRefreshNeeded }) => {
  const [callingPatientPhone, setCallingPatientPhone] = useState(null);
  const [callError, setCallError] = useState('');
  const [callSuccess, setCallSuccess] = useState('');


  const handleTriggerCall = async (patient) => {
    if (!patient.phoneNumber) {
      setCallError(`Phone number missing for ${patient.name}.`);
      return;
    }
    setCallingPatientPhone(patient.phoneNumber);
    setCallError('');
    setCallSuccess('');

    try {
      const response = await fetch(`${apiBaseUrl}/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          to: patient.phoneNumber,
          participant_name: patient.name // Backend expects participant_name
        }),
      });

      const data = await response.json();
      if (!response.ok || !data.sid) {
        throw new Error(data.message || 'Failed to initiate call.');
      }
      
      setCallSuccess(`Call initiated to ${patient.name}. Summary will update after call completion.`);
      // Optionally, refresh data after a delay or use websockets for real-time updates
      setTimeout(() => {
        onRefreshNeeded();
        setCallSuccess(''); // Clear success message after refresh
      }, 30000); // Refresh after 30 seconds (adjust as needed)

    } catch (error) {
      console.error('Error triggering call:', error);
      setCallError(`Error calling ${patient.name}: ${error.message}`);
    } finally {
      setCallingPatientPhone(null);
    }
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
            {patients.map((patient) => (
              <tr key={patient.id || patient.phoneNumber}>
                <td>
                  <div>{patient.name}</div>
                  <div style={{ fontSize: '0.8rem', color: '#546e7a' }}>{patient.phoneNumber}</div>
                </td>
                <td>{patient.lastCallDate ? new Date(patient.lastCallDate).toLocaleDateString() : 'N/A'}</td>
                <td className="call-summary-cell" title={patient.callSummary || 'No summary available.'}>
                  {patient.callSummary || 'No summary available.'}
                </td>
                <td className="actions-cell">
                  <button
                    onClick={() => handleTriggerCall(patient)}
                    disabled={callingPatientPhone === patient.phoneNumber}
                  >
                    {callingPatientPhone === patient.phoneNumber ? 'Calling...' : 'Trigger Call'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
};

export default PatientTableSimple;