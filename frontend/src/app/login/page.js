"use client";
import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import '../../styles/MinimalDashboard.css'; // Import the new CSS

const LoginPage = () => {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const router = useRouter();

  const handleLogin = (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    // IMPORTANT: This is a placeholder for actual authentication.
    // In a real application, this should be a secure call to a backend.
    if (password === 'staff123') { // Example password
      sessionStorage.setItem('isStaffAuthenticated', 'true');
      router.push('/dashboard');
    } else {
      setError('Invalid password. Please try again.');
    }
  };

  return (
    <div className="login-page">
      <div className="login-container">
        {/* You can add a logo here if you have one in public/assets */}
        {/* <img src="/assets/your-logo.png" alt="Logo" className="logo" /> */}
        <h1>Staff Login</h1>
        <form onSubmit={handleLogin} className="login-form">
          <div className="input-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              aria-describedby={error ? "password-error" : undefined}
            />
          </div>
          {error && <p id="password-error" className="error-message">{error}</p>}
          <button type="submit" className="submit-button">Login</button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;