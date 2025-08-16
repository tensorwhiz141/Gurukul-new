import React, { useState, useEffect, useRef } from 'react';
import { useDispatch } from 'react-redux';
import {
  useProcessBHIVRequestMutation,
  useQuickQueryMutation,
  useClassifyQueryMutation,
  useGetBHIVStatsQuery,
  checkBHIVAvailability,
  formatBHIVResponse,
  createQuickBHIVRequest,
  createEducationalBHIVRequest,
  createWellnessBHIVRequest,
  createSpiritualBHIVRequest
} from '../api/bhivCoreApiSlice';
import {
  ClassificationType,
  UrgencyLevel,
  InputMode,
  UserPersona
} from '../types/bhivCoreSchema';

const BHIVCoreInterface = ({ userId = "demo-user", className = "" }) => {
  const dispatch = useDispatch();
  const [query, setQuery] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [selectedMode, setSelectedMode] = useState('quick');
  const [classification, setClassification] = useState(null);
  const [response, setResponse] = useState(null);
  const [isAvailable, setIsAvailable] = useState(false);
  const [sessionHistory, setSessionHistory] = useState([]);
  const responseRef = useRef(null);

  // API hooks
  const [processBHIVRequest, { isLoading: isProcessing }] = useProcessBHIVRequestMutation();
  const [quickQuery, { isLoading: isQuickLoading }] = useQuickQueryMutation();
  const [classifyQuery, { isLoading: isClassifying }] = useClassifyQueryMutation();
  const { data: stats, refetch: refetchStats } = useGetBHIVStatsQuery();

  // Check BHIV availability on mount
  useEffect(() => {
    const checkAvailability = async () => {
      const available = await checkBHIVAvailability(dispatch);
      setIsAvailable(available);
    };
    
    checkAvailability();
  }, [dispatch]);

  // Auto-scroll to response
  useEffect(() => {
    if (response && responseRef.current) {
      responseRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [response]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !isAvailable) return;

    try {
      let result;
      const timestamp = new Date().toISOString();

      if (selectedMode === 'quick') {
        // Quick query mode
        result = await quickQuery({
          userId,
          query: query.trim(),
          voiceEnabled,
          lang: "en-IN"
        }).unwrap();

        // Format quick response
        const formattedResponse = {
          sessionId: result.session_id,
          responseText: result.response_text,
          confidenceScore: result.confidence_score,
          agentUsed: result.agent_used,
          classification: result.classification,
          processingTimeMs: result.processing_time_ms,
          sources: result.sources || [],
          followupSuggestions: result.followup_suggestions || [],
          voiceResponseUrl: result.voice_response_url,
          timestamp
        };

        setResponse(formattedResponse);
        
      } else {
        // Full BHIV processing mode
        let bhivRequest;

        switch (selectedMode) {
          case 'educational':
            bhivRequest = createEducationalBHIVRequest(userId, query.trim(), {
              voiceEnabled,
              lang: "en-IN"
            });
            break;
          case 'wellness':
            bhivRequest = createWellnessBHIVRequest(userId, query.trim(), {
              voiceEnabled,
              lang: "en-IN"
            });
            break;
          case 'spiritual':
            bhivRequest = createSpiritualBHIVRequest(userId, query.trim(), {
              voiceEnabled,
              lang: "en-IN"
            });
            break;
          default:
            bhivRequest = createQuickBHIVRequest(userId, query.trim(), {
              voiceEnabled,
              lang: "en-IN"
            });
        }

        result = await processBHIVRequest(bhivRequest).unwrap();
        const formattedResponse = formatBHIVResponse(result);
        setResponse(formattedResponse);
      }

      // Add to session history
      setSessionHistory(prev => [...prev, {
        id: Date.now(),
        query: query.trim(),
        response: result,
        mode: selectedMode,
        timestamp
      }]);

      // Clear query
      setQuery('');
      
      // Refresh stats
      refetchStats();

    } catch (error) {
      console.error('BHIV query failed:', error);
      setResponse({
        error: true,
        message: error.data?.detail || error.message || 'Failed to process query',
        timestamp: new Date().toISOString()
      });
    }
  };

  const handleClassify = async () => {
    if (!query.trim()) return;

    try {
      const result = await classifyQuery({
        userId,
        query: query.trim(),
        lang: "en-IN"
      }).unwrap();

      setClassification(result);
    } catch (error) {
      console.error('Classification failed:', error);
    }
  };

  const getClassificationColor = (classification) => {
    const colors = {
      [ClassificationType.LEARNING_QUERY]: 'text-blue-600',
      [ClassificationType.WELLNESS_QUERY]: 'text-green-600',
      [ClassificationType.SPIRITUAL_QUERY]: 'text-purple-600',
      [ClassificationType.EMERGENCY]: 'text-red-600',
      [ClassificationType.GENERAL_QUERY]: 'text-gray-600'
    };
    return colors[classification] || 'text-gray-600';
  };

  const getUrgencyColor = (urgency) => {
    const colors = {
      [UrgencyLevel.LOW]: 'text-gray-500',
      [UrgencyLevel.NORMAL]: 'text-blue-500',
      [UrgencyLevel.HIGH]: 'text-orange-500',
      [UrgencyLevel.CRITICAL]: 'text-red-500'
    };
    return colors[urgency] || 'text-gray-500';
  };

  return (
    <div className={`bhiv-core-interface p-6 max-w-4xl mx-auto ${className}`}>
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          BHIV Core Interface
        </h2>
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 ${isAvailable ? 'text-green-600' : 'text-red-600'}`}>
            <div className={`w-3 h-3 rounded-full ${isAvailable ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium">
              {isAvailable ? 'BHIV Core Available' : 'BHIV Core Unavailable'}
            </span>
          </div>
          {stats && (
            <div className="text-sm text-gray-600">
              Sessions: {stats.total_sessions} | Active: {stats.active_sessions}
            </div>
          )}
        </div>
      </div>

      {/* Query Form */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="space-y-4">
          {/* Query Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything... BHIV Core will route your query to the appropriate agent."
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
              disabled={!isAvailable}
            />
          </div>

          {/* Options */}
          <div className="flex flex-wrap gap-4">
            {/* Mode Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Processing Mode
              </label>
              <select
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                disabled={!isAvailable}
              >
                <option value="quick">Quick Query</option>
                <option value="educational">Educational</option>
                <option value="wellness">Wellness</option>
                <option value="spiritual">Spiritual</option>
                <option value="full">Full BHIV Processing</option>
              </select>
            </div>

            {/* Voice Toggle */}
            <div className="flex items-center">
              <label className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <input
                  type="checkbox"
                  checked={voiceEnabled}
                  onChange={(e) => setVoiceEnabled(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  disabled={!isAvailable}
                />
                Voice Response
              </label>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              type="submit"
              disabled={!query.trim() || !isAvailable || isProcessing || isQuickLoading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {(isProcessing || isQuickLoading) && (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              )}
              {selectedMode === 'quick' ? 'Quick Query' : 'Process with BHIV'}
            </button>

            <button
              type="button"
              onClick={handleClassify}
              disabled={!query.trim() || !isAvailable || isClassifying}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isClassifying ? 'Classifying...' : 'Classify Only'}
            </button>
          </div>
        </div>
      </form>

      {/* Classification Result */}
      {classification && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Query Classification</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Type:</span>
              <div className={`font-medium ${getClassificationColor(classification.classification)}`}>
                {classification.classification}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Confidence:</span>
              <div className="font-medium">{(classification.confidence * 100).toFixed(1)}%</div>
            </div>
            <div>
              <span className="text-gray-600">Urgency:</span>
              <div className={`font-medium ${getUrgencyColor(classification.urgency)}`}>
                {classification.urgency}
              </div>
            </div>
            <div>
              <span className="text-gray-600">Agent:</span>
              <div className="font-medium">{classification.recommended_agent}</div>
            </div>
          </div>
          <div className="mt-2">
            <span className="text-gray-600">Reasoning:</span>
            <div className="text-sm text-gray-800">{classification.reasoning}</div>
          </div>
        </div>
      )}

      {/* Response */}
      {response && (
        <div ref={responseRef} className="mb-6 p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
          <h3 className="font-semibold text-gray-800 mb-3">BHIV Response</h3>
          
          {response.error ? (
            <div className="text-red-600">
              <strong>Error:</strong> {response.message}
            </div>
          ) : (
            <div className="space-y-4">
              {/* Response Text */}
              <div className="prose max-w-none">
                <p className="text-gray-800 leading-relaxed">{response.responseText}</p>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600 border-t pt-3">
                <div>
                  <span className="font-medium">Agent:</span> {response.agentUsed}
                </div>
                <div>
                  <span className="font-medium">Confidence:</span> {(response.confidenceScore * 100).toFixed(1)}%
                </div>
                <div>
                  <span className="font-medium">Processing:</span> {response.processingTimeMs?.toFixed(0)}ms
                </div>
                <div>
                  <span className="font-medium">Classification:</span> {response.classification}
                </div>
              </div>

              {/* Follow-up Suggestions */}
              {response.followupSuggestions?.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-2">Follow-up Suggestions:</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                    {response.followupSuggestions.map((suggestion, index) => (
                      <li key={index}>{suggestion}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Sources */}
              {response.sources?.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-2">Sources:</h4>
                  <div className="text-sm text-gray-600">
                    {response.sources.join(', ')}
                  </div>
                </div>
              )}

              {/* Voice Response */}
              {response.voiceResponseUrl && (
                <div>
                  <h4 className="font-medium text-gray-800 mb-2">Voice Response:</h4>
                  <audio controls className="w-full">
                    <source src={response.voiceResponseUrl} type="audio/mpeg" />
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Session History */}
      {sessionHistory.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Session History</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {sessionHistory.slice(-5).reverse().map((session) => (
              <div key={session.id} className="p-3 bg-gray-50 rounded text-sm">
                <div className="font-medium text-gray-800 mb-1">
                  {session.query}
                </div>
                <div className="text-gray-600">
                  Mode: {session.mode} | {new Date(session.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default BHIVCoreInterface;
