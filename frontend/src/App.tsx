import './App.css';
import { useState } from 'react';

interface Message {
  text: string;
  sender: 'user' | 'assistant';
}

function App() {
 const [message, setMessage] =useState("");

 const [messages, setMessages] = useState<Message[]>([]);
 
 const [loading, setLoading] = useState(false);

 const sendMessage = async () =>{
  if(!message.trim() || loading ) return;

  const userMessage = message.trim();
  setMessages(
    (prev)=>[...prev,{text: userMessage, sender: 'user'}]
  );
  setMessage("");
  setLoading(true);

 try{
  //send request to fastapi
  const response = await fetch("http://127.0.0.1:8000/chat",{
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({message: userMessage})
  });
 
  if(!response.ok){
    throw new Error("Network response was not ok");
  }
  // Convert JSON response into JavaScript object
      const data = await response.json();

console.log("API Status:", response.status);
console.log("API Response:", data);
console.log("FULL BACKEND RESPONSE:", data);
console.log("RESPONSE FIELD:", data.response);

setMessages((prev) => [
  ...prev,
  {
    text: data.response || "Response field is missing!",
    sender: "assistant",
  },
]);
    } catch (error) {
      console.error("Backend Error:", error);

  setMessages((prev) => [
    ...prev,
    {
      text: "Error: " + String(error),
      sender: "assistant",
    },
  ]);
    } finally {
      setLoading(false);
    }
 };

 return(
  <div className="app">
      <h1>Mini GPT</h1>
      <p>Your personal AI assistant</p>
      <div className="chat-box">
       {
        messages.map((msg,index) => (
          <div key= {index} className={`${msg.sender}`}>
            {msg.text}
          </div>
        ))
       }
      {loading && <div className="assistant">Thinking...</div>}

      </div>

      <div className="input-area">
     <input value={message} onChange={(e) => setMessage(e.target.value)}
      onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
      placeholder="Type your message..." />
     <button onClick={sendMessage} disabled={loading}>Send</button>
      </div>
    </div>
 );

}

export default App;