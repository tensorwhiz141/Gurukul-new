/**
 * Persona Selector Component
 * 
 * This component provides an interface for selecting and managing different
 * AI personas with their associated memory contexts and capabilities.
 */

import React, { useState, useEffect } from 'react';
import { usePersonaSummary } from '../memoryHooks';

const PERSONA_CONFIG = {
  'financial_advisor': {
    name: 'Financial Advisor',
    description: 'Expert in investment strategies, portfolio management, and financial planning',
    domain: 'finance',
    icon: '💰',
    color: '#2E8B57',
    capabilities: [
      'Investment advice',
      'Portfolio analysis', 
      'Risk assessment',
      'Retirement planning',
      'Tax optimization'
    ],
    specialties: [
      'Conservative investing',
      'Growth strategies',
      'Diversification',
      'Market analysis'
    ]
  },
  'budget_planner': {
    name: 'Budget Planner',
    description: 'Specialist in budgeting, expense tracking, and savings optimization',
    domain: 'finance',
    icon: '📊',
    color: '#4169E1',
    capabilities: [
      'Budget creation',
      'Expense tracking',
      'Savings goals',
      'Debt management',
      'Cash flow analysis'
    ],
    specialties: [
      'Monthly budgeting',
      'Emergency funds',
      'Debt reduction',
      'Expense optimization'
    ]
  },
  'investment_coach': {
    name: 'Investment Coach',
    description: 'Focused on investment education and market guidance',
    domain: 'finance',
    icon: '📈',
    color: '#FF6347',
    capabilities: [
      'Market education',
      'Investment strategies',
      'Stock analysis',
      'Options trading',
      'Crypto guidance'
    ],
    specialties: [
      'Technical analysis',
      'Fundamental analysis',
      'Market timing',
      'Risk management'
    ]
  },
  'gurukul_math_tutor': {
    name: 'Math Tutor',
    description: 'Expert mathematics educator for all levels',
    domain: 'education',
    icon: '🔢',
    color: '#9932CC',
    capabilities: [
      'Algebra tutoring',
      'Calculus help',
      'Geometry lessons',
      'Statistics guidance',
      'Problem solving'
    ],
    specialties: [
      'Step-by-step solutions',
      'Concept explanation',
      'Practice problems',
      'Exam preparation'
    ]
  },
  'gurukul_science_tutor': {
    name: 'Science Tutor',
    description: 'Comprehensive science education across multiple disciplines',
    domain: 'education',
    icon: '🔬',
    color: '#228B22',
    capabilities: [
      'Physics concepts',
      'Chemistry help',
      'Biology tutoring',
      'Lab experiments',
      'Scientific method'
    ],
    specialties: [
      'Conceptual understanding',
      'Practical applications',
      'Experiment design',
      'Theory explanation'
    ]
  },
  'karma_advisor': {
    name: 'Karma Advisor',
    description: 'Spiritual guide for karma understanding and life balance',
    domain: 'spiritual',
    icon: '☯️',
    color: '#FF8C00',
    capabilities: [
      'Karma guidance',
      'Life balance',
      'Spiritual growth',
      'Mindfulness practices',
      'Ethical decisions'
    ],
    specialties: [
      'Karmic understanding',
      'Action consequences',
      'Spiritual development',
      'Inner peace'
    ]
  },
  'ask_vedas': {
    name: 'Vedic Scholar',
    description: 'Ancient wisdom and Vedic knowledge expert',
    domain: 'spiritual',
    icon: '📿',
    color: '#B8860B',
    capabilities: [
      'Vedic wisdom',
      'Ancient texts',
      'Philosophical guidance',
      'Meditation practices',
      'Life principles'
    ],
    specialties: [
      'Sanskrit texts',
      'Philosophical insights',
      'Meditation techniques',
      'Spiritual practices'
    ]
  }
};

const PersonaSelector = ({ 
  currentPersona, 
  onPersonaChange, 
  userId,
  showMemoryStats = true,
  layout = 'grid' // 'grid', 'list', 'compact'
}) => {
  const [selectedDomain, setSelectedDomain] = useState('all');
  const [showDetails, setShowDetails] = useState(false);

  // Get unique domains
  const domains = ['all', ...new Set(Object.values(PERSONA_CONFIG).map(p => p.domain))];

  // Filter personas by domain
  const filteredPersonas = Object.entries(PERSONA_CONFIG).filter(([id, config]) => 
    selectedDomain === 'all' || config.domain === selectedDomain
  );

  return (
    <div className="persona-selector">
      {/* Header */}
      <div className="persona-selector-header">
        <h3>Choose Your AI Assistant</h3>
        
        {/* Domain Filter */}
        <div className="domain-filter">
          <label>Domain:</label>
          <select value={selectedDomain} onChange={(e) => setSelectedDomain(e.target.value)}>
            {domains.map(domain => (
              <option key={domain} value={domain}>
                {domain === 'all' ? 'All Domains' : domain.charAt(0).toUpperCase() + domain.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Layout Toggle */}
        <div className="layout-toggle">
          <button 
            onClick={() => setShowDetails(!showDetails)}
            className={showDetails ? 'active' : ''}
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>
      </div>

      {/* Persona Grid/List */}
      <div className={`persona-grid ${layout}`}>
        {filteredPersonas.map(([personaId, config]) => (
          <PersonaCard
            key={personaId}
            personaId={personaId}
            config={config}
            isSelected={currentPersona === personaId}
            onSelect={onPersonaChange}
            userId={userId}
            showMemoryStats={showMemoryStats}
            showDetails={showDetails}
            layout={layout}
          />
        ))}
      </div>

      {/* Current Persona Info */}
      {currentPersona && (
        <div className="current-persona-info">
          <h4>Current Assistant: {PERSONA_CONFIG[currentPersona]?.name}</h4>
          <p>{PERSONA_CONFIG[currentPersona]?.description}</p>
        </div>
      )}
    </div>
  );
};

// Individual Persona Card Component
const PersonaCard = ({ 
  personaId, 
  config, 
  isSelected, 
  onSelect, 
  userId,
  showMemoryStats,
  showDetails,
  layout 
}) => {
  const { stats, isLoading } = usePersonaSummary(personaId, userId);

  const handleSelect = () => {
    onSelect(personaId);
  };

  return (
    <div 
      className={`persona-card ${layout} ${isSelected ? 'selected' : ''}`}
      onClick={handleSelect}
      style={{ borderColor: config.color }}
    >
      {/* Card Header */}
      <div className="persona-card-header">
        <div className="persona-icon" style={{ backgroundColor: config.color }}>
          {config.icon}
        </div>
        <div className="persona-info">
          <h4 className="persona-name">{config.name}</h4>
          <span className="persona-domain">{config.domain}</span>
        </div>
        {isSelected && <div className="selected-indicator">✓</div>}
      </div>

      {/* Card Description */}
      <div className="persona-description">
        {config.description}
      </div>

      {/* Memory Statistics */}
      {showMemoryStats && (
        <div className="persona-memory-stats">
          {isLoading ? (
            <div className="loading-stats">Loading stats...</div>
          ) : stats ? (
            <div className="stats-summary">
              <div className="stat-item">
                <span className="stat-label">Memories:</span>
                <span className="stat-value">{stats.totalMemories}</span>
              </div>
              {stats.lastInteraction && (
                <div className="stat-item">
                  <span className="stat-label">Last chat:</span>
                  <span className="stat-value">
                    {new Date(stats.lastInteraction).toLocaleDateString()}
                  </span>
                </div>
              )}
              {stats.recentTopics.length > 0 && (
                <div className="recent-topics">
                  <span className="topics-label">Recent topics:</span>
                  <div className="topics-list">
                    {stats.recentTopics.slice(0, 3).map((topic, index) => (
                      <span key={index} className="topic-tag">{topic}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="no-stats">No previous interactions</div>
          )}
        </div>
      )}

      {/* Detailed Information */}
      {showDetails && (
        <div className="persona-details">
          {/* Capabilities */}
          <div className="capabilities-section">
            <h5>Capabilities</h5>
            <ul className="capabilities-list">
              {config.capabilities.map((capability, index) => (
                <li key={index}>{capability}</li>
              ))}
            </ul>
          </div>

          {/* Specialties */}
          <div className="specialties-section">
            <h5>Specialties</h5>
            <div className="specialties-tags">
              {config.specialties.map((specialty, index) => (
                <span key={index} className="specialty-tag">{specialty}</span>
              ))}
            </div>
          </div>

          {/* Memory Categories */}
          {stats?.categories && (
            <div className="memory-categories">
              <h5>Memory Distribution</h5>
              <div className="categories-chart">
                {Object.entries(stats.categories).map(([type, count]) => (
                  <div key={type} className="category-bar">
                    <span className="category-label">{type}</span>
                    <div 
                      className="category-fill" 
                      style={{ 
                        width: `${(count / stats.totalMemories) * 100}%`,
                        backgroundColor: config.color 
                      }}
                    ></div>
                    <span className="category-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Quick Actions */}
      <div className="persona-actions">
        <button 
          className="select-button"
          style={{ backgroundColor: config.color }}
          onClick={handleSelect}
        >
          {isSelected ? 'Currently Active' : 'Select Assistant'}
        </button>
        
        {stats?.totalMemories > 0 && (
          <button 
            className="view-memories-button"
            onClick={(e) => {
              e.stopPropagation();
              // Navigate to memory dashboard for this persona
              console.log('View memories for', personaId);
            }}
          >
            View Memories ({stats.totalMemories})
          </button>
        )}
      </div>
    </div>
  );
};

// Compact Persona Selector for Navigation
export const CompactPersonaSelector = ({ currentPersona, onPersonaChange, userId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const currentConfig = PERSONA_CONFIG[currentPersona];

  return (
    <div className="compact-persona-selector">
      <button 
        className="current-persona-button"
        onClick={() => setIsOpen(!isOpen)}
        style={{ borderColor: currentConfig?.color }}
      >
        <span className="persona-icon">{currentConfig?.icon}</span>
        <span className="persona-name">{currentConfig?.name}</span>
        <span className="dropdown-arrow">{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <div className="persona-dropdown">
          {Object.entries(PERSONA_CONFIG).map(([personaId, config]) => (
            <button
              key={personaId}
              className={`persona-option ${currentPersona === personaId ? 'selected' : ''}`}
              onClick={() => {
                onPersonaChange(personaId);
                setIsOpen(false);
              }}
            >
              <span className="persona-icon">{config.icon}</span>
              <span className="persona-name">{config.name}</span>
              <span className="persona-domain">({config.domain})</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Persona Quick Switch Component
export const PersonaQuickSwitch = ({ currentPersona, onPersonaChange, recentPersonas = [] }) => {
  const getRecentPersonas = () => {
    const recent = recentPersonas.filter(id => id !== currentPersona).slice(0, 3);
    return recent.map(id => ({ id, config: PERSONA_CONFIG[id] }));
  };

  return (
    <div className="persona-quick-switch">
      <span className="quick-switch-label">Quick switch:</span>
      {getRecentPersonas().map(({ id, config }) => (
        <button
          key={id}
          className="quick-switch-button"
          onClick={() => onPersonaChange(id)}
          title={config.name}
        >
          {config.icon}
        </button>
      ))}
    </div>
  );
};

export default PersonaSelector;
