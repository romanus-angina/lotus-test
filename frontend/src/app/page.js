"use client";
import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import '../styles/MinimalDashboard.css'
import NavbarSimple from '../components/NavbarSimple';

const RootPage = () => {
const router = useRouter();

// Optional: If you want to automatically redirect to login if not authenticated
// This is usually handled by protected routes, but for a simple root,
// you might just guide the user.
useEffect(() => {
const isAuthenticated = sessionStorage.getItem('isStaffAuthenticated') === 'true';
if (isAuthenticated) {
// If already logged in, maybe redirect to dashboard
// router.push('/dashboard');
}
}, [router]);

const pageStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 'calc(100vh - 70px)', // Assuming navbar height of \~70px, adjust if needed
    textAlign: 'center',
    padding: '20px',
    backgroundColor: '\#e8eff5', // Consistent with login page background
    marginTop: '70px', // Offset for a fixed/sticky navbar
};

const contentBoxStyle = {
    backgroundColor: '\#ffffff',
    padding: '40px 50px',
    borderRadius: '8px',
    boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)',
    maxWidth: '500px',
};

const titleStyle = {
    fontSize: '1.8rem',
    color: '\#263238',
    marginBottom: '15px',
    fontWeight: '600',
};

const textStyle = {
    fontSize: '1rem',
    color: '\#455a64',
    marginBottom: '30px',
    lineHeight: '1.6',
};

const buttonStyle = {
    backgroundColor: '\#00796b', // Teal
    color: 'white',
    border: 'none',
    padding: '12px 25px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '500',
    textDecoration: 'none', // For Link component
    display: 'inline-block',
    transition: 'background-color 0.2s',
};

return (
    <>
        <NavbarSimple />
        <div style={pageStyle}>
            <div style={contentBoxStyle}>
            {/* You can add a logo here if you have one */}
            {/* \<img src="/assets/your-logo.png" alt="Klix AI" style={{ width: '120px', marginBottom: '25px' }} /\> \*/}
                <h1 style={titleStyle}>Welcome to the Clinical AI Platform</h1>
                <p style={textStyle}>
                This system allows authorized clinical staff to manage patient records
                and initiate AI-powered check-in calls.
                </p>
                <Link href="/login" style={buttonStyle} onMouseOver={(e) => e.currentTarget.style.backgroundColor = '\#004d40'} onMouseOut={(e) => e.currentTarget.style.backgroundColor = '\#00796b'}>
                Proceed to Staff Login
            </Link>
            </div>
        </div>
    </>
);
};

export default RootPage;