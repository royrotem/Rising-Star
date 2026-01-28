import { useState, useRef, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft,
  Send,
  Bot,
  User,
  Loader2,
  Lightbulb,
  AlertTriangle,
  BarChart3
} from 'lucide-react';
import clsx from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  data?: {
    type?: string;
    evidence?: string[];
    relatedData?: Record<string, unknown>;
  };
}

const suggestedQueries = [
  "Why is motor A drawing more current?",
  "Show me all events where battery temp exceeded 40C",
  "What changed in the last 7 days?",
  "Compare current performance to baseline",
  "What sensors should we add for the next version?",
];

export default function Conversation() {
  const { systemId } = useParams();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm your AI engineering assistant. I can help you understand your system's behavior, investigate anomalies, and answer questions about your data. What would you like to know?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate AI response
    await new Promise(resolve => setTimeout(resolve, 1500));

    const queryLower = input.toLowerCase();
    let response: Message;

    if (queryLower.includes('why') && queryLower.includes('current')) {
      response = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Based on my analysis, the increased motor current draw is most likely caused by **firmware update v2.3.1**, which was deployed 3 days ago.

The update modified the motor control PID parameters, resulting in a more aggressive response curve. This increases power consumption by approximately **12%** but may improve response time.

**Key Evidence:**
- Current increase started exactly when firmware was deployed
- Pattern is consistent across all operating conditions  
- No mechanical indicators of degradation

**Recommendation:** Consider rolling back to v2.3.0 if the increased power consumption is unacceptable, or consult with the firmware team about the PID tuning changes.`,
        timestamp: new Date(),
        data: {
          type: 'explanation',
          evidence: [
            'Firmware v2.3.1 deployed on 2024-01-10',
            'Current increase: 12% above baseline',
            'No correlation with mechanical wear indicators',
          ],
        },
      };
    } else if (queryLower.includes('show') || queryLower.includes('find')) {
      response = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I found **23 events** matching your criteria in the last 7 days.

**Summary:**
- Peak temperature: 42.3°C (Jan 12, 14:30)
- Average when above threshold: 41.2°C
- Most common time: 14:00-16:00 (high ambient temp)

The pattern suggests thermal issues during afternoon operations. Would you like me to correlate this with motor load data?`,
        timestamp: new Date(),
        data: {
          type: 'data_query',
          relatedData: {
            total_matches: 23,
            peak_value: 42.3,
            peak_time: '2024-01-12T14:30:00Z',
          },
        },
      };
    } else if (queryLower.includes('sensor') || queryLower.includes('next version')) {
      response = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Based on the blind spots I've identified, here are my recommendations for the next hardware revision:

**Priority 1: 3-axis Accelerometer**
- Location: Motor mount
- Sampling: 1kHz
- Rationale: Enable vibration analysis for early bearing wear detection
- Estimated diagnostic coverage improvement: +25%
- Cost: ~$150

**Priority 2: Humidity Sensor**  
- Location: Electronics bay
- Rationale: Correlate environmental conditions with electrical anomalies
- Cost: ~$25

These additions would significantly improve our ability to diagnose the recurring motor anomalies that we currently cannot fully explain.`,
        timestamp: new Date(),
        data: {
          type: 'recommendation',
        },
      };
    } else {
      response = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I can help you investigate that. Currently, the system is operating at **87.5% health** with one active anomaly (motor current deviation).

Try asking me specific questions like:
- "Why is [parameter] behaving this way?"
- "Show me all events where [condition]"
- "Compare [metric] over the last [period]"
- "What are the engineering margins for [component]?"

What would you like to explore?`,
        timestamp: new Date(),
      };
    }

    setMessages(prev => [...prev, response]);
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestedQuery = (query: string) => {
    setInput(query);
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center gap-4 p-4 border-b border-slate-700 bg-slate-800">
        <Link
          to={'/systems/' + systemId}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-500/10 rounded-lg">
            <Bot className="w-6 h-6 text-primary-500" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">Conversational Chief Engineer</h1>
            <p className="text-sm text-slate-400">Ask questions about your system in natural language</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              'flex gap-3 max-w-4xl',
              message.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            )}
          >
            <div className={clsx(
              'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
              message.role === 'user' ? 'bg-primary-500' : 'bg-slate-700'
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
                : 'bg-slate-800 border border-slate-700'
            )}>
              <div className={clsx(
                'prose prose-sm max-w-none',
                message.role === 'user' ? 'prose-invert' : 'prose-slate prose-invert'
              )}>
                {message.content.split('\n').map((line, i) => (
                  <p key={i} className={clsx(
                    'mb-2 last:mb-0',
                    message.role === 'assistant' && 'text-slate-200'
                  )}>
                    {line.split('**').map((part, j) => 
                      j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                    )}
                  </p>
                ))}
              </div>
              
              {message.data?.evidence && (
                <div className="mt-3 pt-3 border-t border-slate-600">
                  <div className="flex items-center gap-2 text-sm text-slate-400 mb-2">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Supporting Evidence</span>
                  </div>
                  <ul className="space-y-1">
                    {message.data.evidence.map((item, i) => (
                      <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                        <span className="text-primary-400">•</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-3 max-w-4xl">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-slate-700">
              <Bot className="w-5 h-5 text-primary-400" />
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-4">
              <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Queries */}
      {messages.length <= 2 && (
        <div className="px-4 pb-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-slate-400">
            <Lightbulb className="w-4 h-4" />
            <span>Suggested questions</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestedQueries.map((query, i) => (
              <button
                key={i}
                onClick={() => handleSuggestedQuery(query)}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
              >
                {query}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-slate-700 bg-slate-800">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your system..."
            className="flex-1 px-4 py-3 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition-colors"
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
