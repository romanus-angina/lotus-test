"use client";
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import NavbarSimple from '../../components/NavbarSimple';
import PatientTableSimple from '../../components/PatientTableSimple';
import AddPatientModalSimple from '../../components/AddPatientModalSimple';
import '../../styles/MinimalDashboard.css'; // Import the new CSS

const DashboardPage = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const router = useRouter();

  // Define API_BASE_URL - replace with your actual backend URL if different
  const API_BASE_URL = 'https://lotustest.ngrok.io';

  const fetchPatientData = async () => {
    setIsLoading(true);
    setError('');
    try {
      // Your backend's /api/calls endpoint returns call logs.
      // We need to process this to create a "patient" list.
      const response = await fetch(`${API_BASE_URL}/api/calls`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to fetch data: ${response.status}`);
      }
      const data = await response.json();

      // Process call data to create a patient-centric view
      const patientMap = new Map();
      if (data.calls && Array.isArray(data.calls)) {
        data.calls.forEach(call => {
          const phoneNumber = call.to_number;
          if (!phoneNumber) return; // Skip calls without a 'to_number'

          const patientName = call.contact_name || call.participant_name || 'Unknown Patient';
          const callDate = call.created_at ? new Date(call.created_at) : null;
          // Use clinical_summary if available, otherwise ai_summary or default
          const summary = call.clinical_summary || call.ai_summary || (call.call_status === 'completed' && !call.transcript ? 'Call completed, no summary.' : 'No summary yet.');


          if (!patientMap.has(phoneNumber) || (callDate && callDate > (patientMap.get(phoneNumber).lastCallDate || 0))) {
            patientMap.set(phoneNumber, {
              id: call._id || call.stream_sid, // Use a unique ID from the call
              name: patientName,
              phoneNumber: phoneNumber,
              lastCallDate: callDate,
              callSummary: summary,
              // Store the raw call for potential future use, e.g., viewing details
              rawCallData: call,
            });
          } else if (patientMap.has(phoneNumber) && !patientMap.get(phoneNumber).name && patientName !== 'Unknown Patient') {
            // If existing entry has no name, but this call does, update it
             patientMap.get(phoneNumber).name = patientName;
          }
        });
      }
      
      const processedPatients = Array.from(patientMap.values()).sort((a,b) => (b.lastCallDate || 0) - (a.lastCallDate || 0));
      setPatients(processedPatients);

    } catch (err) {
      console.error("Error fetching patient data:", err);
      setError(err.message);
      setPatients([]); // Clear patients on error
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const authStatus = sessionStorage.getItem('isStaffAuthenticated');
    if (authStatus === 'true') {
      setIsAuthenticated(true);
      fetchPatientData();
    } else {
      router.push('/login');
    }
  }, [router]); // Removed refreshKey as fetchPatientData is called directly now

  const handleAddPatientSuccess = () => {
    setShowAddPatientModal(false);
    fetchPatientData(); // Refresh patient list after adding a new one
  };

  if (!isAuthenticated) {
    // This will be brief as the redirect should happen quickly.
    // You could show a full-page loader here.
    return <div className="loading-state">Redirecting to login...</div>;
  }

  return (
    <>
      <NavbarSimple />
      <main className="dashboard-main">
        <div className="dashboard-header">
          <h1>Patient Dashboard</h1>
          <button onClick={() => setShowAddPatientModal(true)} className="add-patient-btn">
            + Add Patient
          </button>
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
