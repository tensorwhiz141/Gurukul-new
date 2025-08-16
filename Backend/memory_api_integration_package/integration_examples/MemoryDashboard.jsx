/**
 * Memory Dashboard Component
 * 
 * This component provides a comprehensive view of stored memories,
 * persona statistics, and memory management capabilities.
 */

import React, { useState, useMemo } from 'react';
import { 
  usePersonaMemories, 
  usePersonaSummary, 
  useMemorySearch,
  useMemoryStorage 
} from '../memoryHooks';
import { 
  useUpdateMemoryMutation, 
  useDeleteMemoryMutation,
  CONTENT_TYPES, 
  IMPORTANCE_LEVELS 
} from '../memoryApiSlice';

const MemoryDashboard = ({ userId, personaId, onPersonaChange }) => {
  // State
  const [selectedContentType, setSelectedContentType] = useState('all');
  const [selectedImportance, setSelectedImportance] = useState('all');
  const [editingMemory, setEditingMemory] = useState(null);
  const [showAddMemory, setShowAddMemory] = useState(false);

  // Hooks
  const { memories, memoriesByType, isLoading: memoriesLoading } = usePersonaMemories(
    personaId, 
    userId, 
    { 
      limit: 50,
      contentTypes: selectedContentType !== 'all' ? [selectedContentType] : [],
      minImportance: selectedImportance !== 'all' ? parseInt(selectedImportance) : null
    }
  );
  
  const { stats, isLoading: statsLoading } = usePersonaSummary(personaId, userId);
  const { search, searchResults, searchQuery, clearSearch } = useMemorySearch({ personaId, userId });
  const { storeUserMemory } = useMemoryStorage();
  
  const [updateMemory] = useUpdateMemoryMutation();
  const [deleteMemory] = useDeleteMemoryMutation();

  // Computed values
  const filteredMemories = useMemo(() => {
    if (searchQuery) return searchResults;
    return memories;
  }, [memories, searchResults, searchQuery]);

  const memoryStats = useMemo(() => {
    if (!stats) return null;
    
    return {
      total: stats.totalMemories,
      byType: stats.categories,
      byImportance: stats.importanceDistribution,
      recentTopics: stats.recentTopics
    };
  }, [stats]);

  // Handlers
  const handleSearch = (query) => {
    if (query.trim()) {
      search(query);
    } else {
      clearSearch();
    }
  };

  const handleEditMemory = (memory) => {
    setEditingMemory(memory);
  };

  const handleUpdateMemory = async (memoryId, updateData) => {
    try {
      await updateMemory({ memoryId, updateData }).unwrap();
      setEditingMemory(null);
    } catch (error) {
      console.error('Failed to update memory:', error);
      alert('Failed to update memory: ' + error.message);
    }
  };

  const handleDeleteMemory = async (memoryId) => {
    if (window.confirm('Are you sure you want to delete this memory?')) {
      try {
        await deleteMemory({ memoryId }).unwrap();
      } catch (error) {
        console.error('Failed to delete memory:', error);
        alert('Failed to delete memory: ' + error.message);
      }
    }
  };

  if (memoriesLoading || statsLoading) {
    return <div className="loading">Loading memory dashboard...</div>;
  }

  return (
    <div className="memory-dashboard">
      {/* Dashboard Header */}
      <div className="dashboard-header">
        <h2>Memory Dashboard</h2>
        <div className="persona-selector">
          <label>Persona:</label>
          <select value={personaId} onChange={(e) => onPersonaChange(e.target.value)}>
            <option value="financial_advisor">Financial Advisor</option>
            <option value="budget_planner">Budget Planner</option>
            <option value="gurukul_math_tutor">Math Tutor</option>
            <option value="karma_advisor">Karma Advisor</option>
          </select>
        </div>
      </div>

      {/* Memory Statistics */}
      {memoryStats && (
        <div className="memory-stats">
          <h3>Statistics</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <h4>Total Memories</h4>
              <div className="stat-value">{memoryStats.total}</div>
            </div>
            
            <div className="stat-card">
              <h4>By Type</h4>
              <div className="stat-breakdown">
                {Object.entries(memoryStats.byType).map(([type, count]) => (
                  <div key={type} className="stat-item">
                    <span className="stat-label">{type}:</span>
                    <span className="stat-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="stat-card">
              <h4>Recent Topics</h4>
              <div className="topics-list">
                {memoryStats.recentTopics.map((topic, index) => (
                  <span key={index} className="topic-tag">{topic}</span>
                ))}
              </div>
            </div>
            
            <div className="stat-card">
              <h4>Importance Distribution</h4>
              <div className="importance-chart">
                {Object.entries(memoryStats.byImportance).map(([level, count]) => (
                  <div key={level} className="importance-bar">
                    <span className="importance-label">{level}</span>
                    <div 
                      className="importance-fill" 
                      style={{ width: `${(count / memoryStats.total) * 100}%` }}
                    ></div>
                    <span className="importance-count">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search and Filters */}
      <div className="memory-controls">
        <div className="search-section">
          <input
            type="text"
            placeholder="Search memories..."
            onChange={(e) => handleSearch(e.target.value)}
            className="search-input"
          />
          {searchQuery && (
            <button onClick={clearSearch} className="clear-search">
              Clear Search
            </button>
          )}
        </div>

        <div className="filters-section">
          <select 
            value={selectedContentType} 
            onChange={(e) => setSelectedContentType(e.target.value)}
          >
            <option value="all">All Types</option>
            {Object.values(CONTENT_TYPES).map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>

          <select 
            value={selectedImportance} 
            onChange={(e) => setSelectedImportance(e.target.value)}
          >
            <option value="all">All Importance</option>
            {Object.entries(IMPORTANCE_LEVELS).map(([name, value]) => (
              <option key={value} value={value}>
                {name.replace('_', ' ')} ({value}+)
              </option>
            ))}
          </select>
        </div>

        <button 
          onClick={() => setShowAddMemory(true)} 
          className="add-memory-btn"
        >
          + Add Memory
        </button>
      </div>

      {/* Search Results Info */}
      {searchQuery && (
        <div className="search-info">
          Found {searchResults.length} memories for "{searchQuery}"
        </div>
      )}

      {/* Memory List */}
      <div className="memory-list">
        {filteredMemories.length === 0 ? (
          <div className="no-memories">
            {searchQuery ? 'No memories found for your search.' : 'No memories stored yet.'}
          </div>
        ) : (
          filteredMemories.map(memory => (
            <MemoryCard
              key={memory.memory_id}
              memory={memory}
              onEdit={handleEditMemory}
              onDelete={handleDeleteMemory}
              isEditing={editingMemory?.memory_id === memory.memory_id}
              onUpdate={handleUpdateMemory}
              onCancelEdit={() => setEditingMemory(null)}
            />
          ))
        )}
      </div>

      {/* Add Memory Modal */}
      {showAddMemory && (
        <AddMemoryModal
          userId={userId}
          personaId={personaId}
          onClose={() => setShowAddMemory(false)}
          onAdd={storeUserMemory}
        />
      )}
    </div>
  );
};

// Memory Card Component
const MemoryCard = ({ 
  memory, 
  onEdit, 
  onDelete, 
  isEditing, 
  onUpdate, 
  onCancelEdit 
}) => {
  const [editContent, setEditContent] = useState(memory.content);
  const [editImportance, setEditImportance] = useState(memory.metadata.importance);
  const [editTags, setEditTags] = useState(memory.metadata.tags.join(', '));

  const handleSaveEdit = () => {
    onUpdate(memory.memory_id, {
      content: editContent,
      metadata: {
        ...memory.metadata,
        importance: editImportance,
        tags: editTags.split(',').map(tag => tag.trim()).filter(Boolean)
      }
    });
  };

  if (isEditing) {
    return (
      <div className="memory-card editing">
        <div className="memory-edit-form">
          <textarea
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            className="edit-content"
          />
          
          <div className="edit-metadata">
            <label>
              Importance:
              <select 
                value={editImportance} 
                onChange={(e) => setEditImportance(parseInt(e.target.value))}
              >
                {Object.entries(IMPORTANCE_LEVELS).map(([name, value]) => (
                  <option key={value} value={value}>
                    {name.replace('_', ' ')} ({value})
                  </option>
                ))}
              </select>
            </label>
            
            <label>
              Tags:
              <input
                type="text"
                value={editTags}
                onChange={(e) => setEditTags(e.target.value)}
                placeholder="tag1, tag2, tag3"
              />
            </label>
          </div>
          
          <div className="edit-actions">
            <button onClick={handleSaveEdit} className="save-btn">Save</button>
            <button onClick={onCancelEdit} className="cancel-btn">Cancel</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-card">
      <div className="memory-header">
        <span className={`memory-type ${memory.content_type}`}>
          {memory.content_type}
        </span>
        <span className="memory-importance">
          {'★'.repeat(Math.min(memory.metadata.importance, 5))} 
          {memory.metadata.importance}/10
        </span>
        <span className="memory-date">
          {new Date(memory.timestamp).toLocaleDateString()}
        </span>
      </div>
      
      <div className="memory-content">
        {memory.content}
      </div>
      
      <div className="memory-metadata">
        {memory.metadata.tags.length > 0 && (
          <div className="memory-tags">
            {memory.metadata.tags.map((tag, index) => (
              <span key={index} className="tag">{tag}</span>
            ))}
          </div>
        )}
        
        {memory.metadata.topic && (
          <div className="memory-topic">
            Topic: {memory.metadata.topic}
          </div>
        )}
      </div>
      
      <div className="memory-actions">
        <button onClick={() => onEdit(memory)} className="edit-btn">
          ✏️ Edit
        </button>
        <button onClick={() => onDelete(memory.memory_id)} className="delete-btn">
          🗑️ Delete
        </button>
      </div>
    </div>
  );
};

// Add Memory Modal Component
const AddMemoryModal = ({ userId, personaId, onClose, onAdd }) => {
  const [content, setContent] = useState('');
  const [contentType, setContentType] = useState(CONTENT_TYPES.TEXT);
  const [importance, setImportance] = useState(IMPORTANCE_LEVELS.AVERAGE);
  const [tags, setTags] = useState('');
  const [topic, setTopic] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!content.trim()) return;

    try {
      await onAdd({
        userId,
        personaId,
        content: content.trim(),
        contentType,
        tags: tags.split(',').map(tag => tag.trim()).filter(Boolean),
        importance,
        topic: topic.trim() || null,
        source: 'manual_entry'
      });
      
      onClose();
    } catch (error) {
      console.error('Failed to add memory:', error);
      alert('Failed to add memory: ' + error.message);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Add New Memory</h3>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Content:</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter memory content..."
              required
              rows={4}
            />
          </div>
          
          <div className="form-row">
            <div className="form-group">
              <label>Type:</label>
              <select value={contentType} onChange={(e) => setContentType(e.target.value)}>
                {Object.values(CONTENT_TYPES).map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>
            
            <div className="form-group">
              <label>Importance:</label>
              <select value={importance} onChange={(e) => setImportance(parseInt(e.target.value))}>
                {Object.entries(IMPORTANCE_LEVELS).map(([name, value]) => (
                  <option key={value} value={value}>
                    {name.replace('_', ' ')} ({value})
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="form-group">
            <label>Tags (comma-separated):</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="tag1, tag2, tag3"
            />
          </div>
          
          <div className="form-group">
            <label>Topic:</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Optional topic"
            />
          </div>
          
          <div className="modal-actions">
            <button type="submit" className="add-btn">Add Memory</button>
            <button type="button" onClick={onClose} className="cancel-btn">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default MemoryDashboard;
