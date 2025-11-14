import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Send, Loader, User, Zap, BookOpen, Bot, X, MessageCircle } from 'lucide-react'; 

const API_ENDPOINT = 'http://localhost:8000/query';

/**
 * Message Bubble component for chat history
 */
const MessageBubble = ({ message }) => {
  const isAgent = message.role === 'agent';
  
  return (
    <div className={`flex flex-col max-w-xl mb-4 ${isAgent ? 'items-start' : 'items-end'}`}>
      <div className={`p-4 rounded-2xl shadow-lg backdrop-blur-xl border transition-all duration-300 hover:scale-[1.02] ${
        isAgent 
          ? 'bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-cyan-500/30 text-white' 
          : 'bg-gradient-to-br from-cyan-500 to-blue-500 border-cyan-400/50 text-white'
      }`}>
        <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
      </div>

      {isAgent && message.sources && message.sources.length > 0 && (
        <div className="mt-2 px-4 py-2 bg-slate-800/50 backdrop-blur-xl border border-cyan-500/20 text-xs text-slate-300 rounded-xl shadow-lg">
          <h4 className="font-semibold flex items-center mb-2 text-cyan-400">
            <BookOpen className="w-3 h-3 mr-1" />
            Sources
          </h4>
          <ul className="list-disc ml-4 space-y-1">
            {message.sources.map((source, index) => (
              <li key={index}>
                <span title={source.uri} className="hover:text-cyan-400 transition-colors cursor-pointer">
                  {source.title}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

/**
 * Main AI Chatbot component for portfolio
 */
const PortfolioAIChatbot = () => {
  const [query, setQuery] = useState('');
  const [messageHistory, setMessageHistory] = useState([
    {
      role: 'agent',
      text: "Welcome! I'm Rishabh's AI Portfolio Assistant. I can answer questions about his skills, projects, and professional background, speaking from his point of view. Ask me anything!"
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isOpen, setIsOpen] = useState(false);

  const messagesEndRef = useRef(null);

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messageHistory]);

  /**
   * Handles the form submission and communicates with the FastAPI backend.
   */
  const handleQuery = useCallback(async (e) => {
    if (e) e.preventDefault();
    const userQuery = query.trim();
    if (!userQuery) return;

    // 1. Add user message to history
    setMessageHistory(prev => [...prev, { role: 'user', text: userQuery }]);
    setQuery('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userQuery }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // 2. Add agent message to history
      setMessageHistory(prev => [
        ...prev, 
        { 
          role: 'agent', 
          text: data.answer, 
          sources: data.sources || [] 
        }
      ]);
      
    } catch (err) {
      console.error("Fetch Error:", err);
      setError(`Failed to get answer: ${err.message}. Make sure the backend server (FastAPI) is running.`);
      setMessageHistory(prev => [...prev, { role: 'agent', text: `ERROR: ${err.message}`, sources: [] }]);
    } finally {
      setIsLoading(false);
    }
  }, [query]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery(e);
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-8 right-8 z-50 w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full shadow-2xl shadow-cyan-500/50 hover:shadow-cyan-500/80 transition-all duration-300 transform hover:scale-110 flex items-center justify-center group"
        style={{ animation: 'glow 3s infinite' }}
      >
        {isOpen ? (
          <X className="text-white" size={28} />
        ) : (
          <MessageCircle className="text-white group-hover:scale-110 transition-transform duration-300" size={28} />
        )}
      </button>

      <style>{`
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.5); }
          50% { box-shadow: 0 0 40px rgba(6, 182, 212, 0.8), 0 0 60px rgba(6, 182, 212, 0.4); }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.3);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(6, 182, 212, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(6, 182, 212, 0.8);
        }
      `}</style>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-28 right-8 z-50 w-full max-w-lg h-[600px] bg-gradient-to-br from-slate-900/95 to-slate-950/95 backdrop-blur-xl shadow-2xl rounded-2xl border border-cyan-500/20 flex flex-col overflow-hidden transition-all duration-300 transform">
          
          {/* Header */}
          <header className="flex items-center justify-between p-4 bg-gradient-to-r from-slate-800/80 to-slate-900/80 backdrop-blur-xl border-b border-cyan-500/20 shadow-lg">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Bot className="w-8 h-8 text-cyan-400 animate-pulse" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-slate-900"></div>
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">AI Portfolio Assistant</h3>
                <p className="text-xs text-cyan-400">Powered by RAG Technology</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-slate-400 hover:text-cyan-400 transition-colors p-2 hover:bg-slate-800/50 rounded-lg"
            >
              <X size={20} />
            </button>
          </header>

          {/* Message History Area */}
          <div className="flex-grow p-4 space-y-4 overflow-y-auto bg-slate-950/50 custom-scrollbar">
            {messageHistory.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))}
            
            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-center self-start text-cyan-400 p-3 rounded-xl bg-slate-800/50 backdrop-blur-xl border border-cyan-500/30 shadow-lg max-w-xs">
                <Loader className="w-4 h-4 mr-2 animate-spin" />
                <span className="text-sm">Thinking...</span>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-500/20 border border-red-500/50 text-red-300 rounded-xl text-sm font-medium backdrop-blur-xl">
                Backend Error: {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Bar */}
          <footer className="p-4 border-t border-cyan-500/20 bg-gradient-to-r from-slate-800/80 to-slate-900/80 backdrop-blur-xl">
            <div className="flex space-x-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about skills, projects, experience..."
                className="flex-grow p-3 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/50 transition-all duration-300"
                disabled={isLoading}
              />
              <button
                onClick={handleQuery}
                className="px-5 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-xl shadow-lg shadow-cyan-500/50 hover:shadow-cyan-500/80 transition-all duration-300 disabled:opacity-50 flex items-center justify-center transform hover:scale-105 hover:-translate-y-0.5 group"
                disabled={isLoading || !query.trim()}
                title="Send Message"
              >
                <Send className="w-5 h-5 group-hover:translate-x-0.5 transition-transform duration-300" />
              </button>
            </div>
          </footer>
        </div>
      )}
    </>
  );
};

export default PortfolioAIChatbot;