"use client";
import React from 'react';
import { useRouter } from 'next/navigation';
import '../styles/MinimalDashboard.css'; // Using styles from here

const NavbarSimple = () => {
  const router = useRouter();

  const handleLogout = () => {
    sessionStorage.removeItem('isStaffAuthenticated');
    router.push('/login');
  };

  return (
    <nav className="dashboard-navbar">
      <div className="logo-area">
        {/* <img src="/assets/your-logo.png" alt="Logo" className="logo" /> */}
        <span className="app-title">Clinical Dashboard</span>
      </div>
      <button onClick={handleLogout} className="logout-button">Logout</button>
    </nav>
  );
};

export default NavbarSimple;