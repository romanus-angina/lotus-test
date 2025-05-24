"use client";
import React from 'react';
import '../styles/MinimalDashboard.css';

const TranscriptModal = ({ patient, onClose }) => {
  if (!patient) return null;

  const formatTranscript = (transcript) => {
    if (!transcript) return "No transcript available.";
    
    // Split by speaker and format nicely
    return transcript.split('\n').map((line, index) => {
      if (line.trim() === '') return null;
      
      const isPatient = line.startsWith('PATIENT:');
      const isLotus = line.startsWith('LOTUS_AI:');
      
      if (isPatient || isLotus) {
        const speaker = isPatient ? 'PATIENT' : 'LOTUS AI';
        const content = line.replace(/^(PATIENT:|LOTUS_AI:)\s*/, '');
        
        return (
          <div key={index} className={`transcript-line ${isPatient ? 'patient' : 'ai'}`}>
            <strong>{speaker}:</strong> {content}
          </div>
        );
      }
      
      return <div key={index} className="transcript-line">{line}</div>;
    }).filter(Boolean);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content transcript-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Call Transcript - {patient.name}</h2>
          <button onClick={onClose} className="modal-close-btn">&times;</button>
        </div>
        
        <div className="transcript-info">
          <p><strong>Phone:</strong> {patient.phoneNumber}</p>
          <p><strong>Last Call:</strong> {patient.lastCallDate ? new Date(patient.lastCallDate).toLocaleString() : 'N/A'}</p>
          <p><strong>Status:</strong> {patient.summaryStatus || 'Unknown'}</p>
        </div>
        
        <div className="transcript-content">
          <h3>Call Summary:</h3>
          <div className="summary-box">
            {patient.callSummary || "No summary available yet."}
          </div>
          
          <h3>Full Transcript:</h3>
          <div className="transcript-box">
            {patient.hasTranscript ? (
              <div className="transcript-lines">
                {formatTranscript(patient.transcript)}
              </div>
            ) : (
              <p className="no-transcript">No transcript available for this call.</p>
            )}
          </div>
        </div>
        
        <div className="modal-actions">
          <button onClick={onClose} className="modal-button primary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default TranscriptModal;
