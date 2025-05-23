import { Inter } from 'next/font/google'
import './globals.css' // Global styles

// If you intend to use the Inter font:
const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Clinical AI Dashboard',
  description: 'Patient management and AI-powered call system for clinical staff.',
  icons: {
    icon: '/favicon.ico', // Ensure favicon.ico is in frontend/public/
  },
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div id="app-root">
          {children}
        </div>
      </body>
    </html>
  )
}