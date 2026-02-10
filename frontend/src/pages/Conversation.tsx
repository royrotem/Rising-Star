import { useState, useRef, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Send,
  Bot,
  User,
  Loader2,
  Lightbulb,
  AlertTriangle,
  Trash2,
  Sparkles,
  Zap,
} from 'lucide-react';
import clsx from 'clsx';
import { chatApi, ChatMessageResponse } from '../services/chatApi';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  aiPowered?: boolean;
  data?: {
    type?: string;
    evidence?: string[];
    relatedData?: Record<string, unknown>;
  };
}

const suggestedQueries = [
  "What anomalies have been detected recently?",
  "Summarize the overall system health",
  "Which fields show the most variance?",
  "Are there any correlations between anomalies?",
  "What recommendations do you have for this system?",
];

const WELCOME_MESSAGE: Message = {
  id: 'welcome',
  role: 'assistant',
  content: "Hello! I'm your AI engineering assistant. I can help you understand your system's behavior, investigate anomalies, and answer questions about your data. What would you like to know?",
  timestamp: new Date(),
  aiPowered: true,
};

export default function Conversation() {
  const { systemId } = useParams();
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [clearing, setClearing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversation history on mount
  useEffect(() => {
    if (!systemId || historyLoaded) return;

    const loadHistory = async () => {
      try {
        const history = await chatApi.history(systemId);
        if (history.messages && history.messages.length > 0) {
          const restored: Message[] = history.messages.map((m) => ({
            id: m.id,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: new Date(m.timestamp),
            data: m.data as Message['data'],
          }));
          setMessages(restored);
        }
      } catch {
        // History not available — keep welcome message
      } finally {
        setHistoryLoaded(true);
      }
    };
    loadHistory();
  }, [systemId, historyLoaded]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading || !systemId) return;

    const userText = input.trim();
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userText,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response: ChatMessageResponse = await chatApi.send(systemId, userText);

      const assistantMessage: Message = {
        id: response.id,
        role: 'assistant',
        content: response.content,
        timestamp: new Date(response.timestamp),
        aiPowered: response.ai_powered,
        data: response.data as Message['data'],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error processing your request. Please try again.\n\n_Error: ${errorMsg}_`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, systemId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestedQuery = (query: string) => {
    setInput(query);
  };

  const handleClearHistory = async () => {
    if (!systemId || clearing) return;
    setClearing(true);
    try {
      await chatApi.clear(systemId);
      setMessages([WELCOME_MESSAGE]);
    } catch {
      // Ignore clear errors
    } finally {
      setClearing(false);
    }
  };

  const hasConversation = messages.length > 1 || (messages.length === 1 && messages[0].id !== 'welcome');

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center gap-4 p-4 border-b border-stone-600 bg-stone-700/80 backdrop-blur-sm">
        <Link
          to={'/systems/' + systemId}
          className="p-2 hover:bg-stone-600 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-stone-400" />
        </Link>
        <div className="flex items-center gap-3 flex-1">
          <div className="p-2 bg-primary-500/10 rounded-lg">
            <Bot className="w-6 h-6 text-primary-500" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white flex items-center gap-2">
              Conversational Chief Engineer
              <Sparkles className="w-4 h-4 text-amber-400" />
            </h1>
            <p className="text-sm text-stone-400 flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-emerald-400" />
              AI-powered — ask questions about your system in natural language
            </p>
          </div>
        </div>
        {hasConversation && (
          <button
            onClick={handleClearHistory}
            disabled={clearing}
            className="p-2 hover:bg-stone-600 rounded-lg transition-colors text-stone-400 hover:text-red-400"
            title="Clear conversation"
          >
            {clearing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Trash2 className="w-5 h-5" />
            )}
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              'flex gap-3 max-w-4xl animate-fadeIn',
              message.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            )}
          >
            <div className={clsx(
              'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
              message.role === 'user' ? 'bg-primary-500' : 'bg-stone-700'
            )}>
              {message.role === 'user' ? (
                <User className="w-5 h-5 text-white" />
              ) : (
                <Bot className="w-5 h-5 text-primary-400" />
              )}
            </div>
            <div className={clsx(
              'rounded-xl p-4 max-w-2xl',
              message.role === 'user'
                ? 'bg-primary-500 text-white'
                : 'bg-stone-700 border border-stone-600'
            )}>
              <div className={clsx(
                'prose prose-sm max-w-none',
                message.role === 'user' ? 'prose-invert' : 'prose-slate prose-invert'
              )}>
                {message.content.split('\n').map((line, i) => (
                  <p key={i} className={clsx(
                    'mb-2 last:mb-0',
                    message.role === 'assistant' && 'text-stone-200'
                  )}>
                    {line.split('**').map((part, j) =>
                      j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                    )}
                  </p>
                ))}
              </div>

              {message.data?.evidence && (
                <div className="mt-3 pt-3 border-t border-stone-600">
                  <div className="flex items-center gap-2 text-sm text-stone-400 mb-2">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Supporting Evidence</span>
                  </div>
                  <ul className="space-y-1">
                    {message.data.evidence.map((item, i) => (
                      <li key={i} className="text-sm text-stone-300 flex items-start gap-2">
                        <span className="text-primary-400">•</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* AI-powered badge for assistant messages */}
              {message.role === 'assistant' && message.aiPowered && message.id !== 'welcome' && (
                <div className="mt-2 pt-2 border-t border-stone-600/50">
                  <span className="inline-flex items-center gap-1 text-xs text-emerald-400/70">
                    <Sparkles className="w-3 h-3" />
                    AI-powered response
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-3 max-w-4xl animate-fadeIn">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-stone-700">
              <Bot className="w-5 h-5 text-primary-400" />
            </div>
            <div className="bg-stone-700 border border-stone-600 rounded-xl p-4">
              <div className="flex items-center gap-2">
                <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
                <span className="text-sm text-stone-400">Analyzing your data...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Queries */}
      {messages.length <= 2 && (
        <div className="px-4 pb-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-stone-400">
            <Lightbulb className="w-4 h-4" />
            <span>Suggested questions</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestedQueries.map((query, i) => (
              <button
                key={i}
                onClick={() => handleSuggestedQuery(query)}
                className="px-3 py-1.5 bg-stone-700 hover:bg-stone-600 border border-stone-600 rounded-lg text-sm text-stone-300 transition-colors"
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-stone-600 bg-stone-700/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your system..."
            className="flex-1 px-4 py-3 bg-stone-700 border border-stone-600 rounded-xl text-white placeholder-stone-400 focus:outline-none focus:border-primary-500 transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-4 py-3 bg-primary-500 hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
