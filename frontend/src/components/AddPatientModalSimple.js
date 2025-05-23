"use client";
import React, { useState } from 'react';
import '../styles/MinimalDashboard.css'; // Using styles from here

const AddPatientModalSimple = ({ onClose, onAddSuccess, apiBaseUrl }) => {
  const [name, setName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    if (!name.trim() || !phoneNumber.trim()) {
      setError('Both name and phone number are required.');
      setIsLoading(false);
      return;
    }

    if (!phoneNumber.startsWith('+')) {
        setError("Phone number must start with a country code (e.g., +1).");
        setIsLoading(false);
        return;
    }

    try {
      // API call to add contact (patient)
      // Your backend uses /api/contacts for this
      const response = await fetch(`${apiBaseUrl}/api/contacts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          phone_number: phoneNumber,
          // study_id: "default_study" // If your backend requires it
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add patient');
      }
      
      // const newPatientData = await response.json(); // If backend returns the new patient
      onAddSuccess(); // This will trigger a refresh in DashboardPage
      onClose(); // Close the modal

    } catch (err) {
      console.error('Error adding patient:', err);
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Add New Patient</h2>
          <button onClick={onClose} className="modal-close-btn">&times;</button>
        </div>
        <form onSubmit={handleSubmit} className="modal-form">
          <div className="input-group">
            <label htmlFor="patientName">Patient Name</label>
            <input
              id="patientName"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <label htmlFor="patientPhone">Phone Number</label>
            <input
              id="patientPhone"
              type="tel"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
              placeholder="+12345678900"
              required
            />
            <small>Include country code (e.g., +1 for US).</small>
          </div>

          {error && <p className="error-message" style={{textAlign: 'left', marginTop: '10px'}}>{error}</p>}

          <div className="modal-actions">
            <button type="button" onClick={onClose} className="modal-button secondary">
              Cancel
            </button>
            <button type="submit" className="modal-button primary" disabled={isLoading}>
              {isLoading ? 'Adding...' : 'Add Patient'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddPatientModalSimple;