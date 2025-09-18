import { useState } from 'react'
import ChatWidget from './ChatWidget'

function App() {
  

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#f3f4f6" }}>
      <div style={{ width: 600, textAlign: "center" }}>
        <h1>UCC Chatbot Demo</h1>
        <ChatWidget headerTitle="Support Chat" />
      </div>
    </div>
  )
}

export default App
