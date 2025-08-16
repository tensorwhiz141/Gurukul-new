# Avatar Chat Feature Documentation

## Overview

The Avatar Chat feature provides an interactive AI assistant that appears as a floating 3D avatar with contextual chat capabilities. Users can double-click the pinned avatar to activate a minimal chat interface that provides contextually aware responses based on the current page content.

## Features

### 🎯 **Trigger Mechanism**
- **Double-click activation**: Double-click the floating pinned avatar to open/close the chat interface
- **Clean design**: No visual indicators or distracting elements on the avatar
- **Simple tooltip**: Minimal "Double-click to chat" guidance on hover

### 💬 **Chat Interface Design**
- **Minimal terminal-inspired UI**: Clean interface with terminal functionality but Gurukul aesthetics
- **Glass morphism styling**: Consistent with platform design using backdrop blur and transparency
- **Terminal prompts**: Monospace text with terminal prompts (> for user, $ for AI) in platform colors
- **Typewriter animations**: Character-by-character typing effects for AI responses
- **Smart positioning**: Automatically positions near the avatar with screen boundary detection
- **Click-outside dismissal**: Closes when clicking anywhere outside the interface
- **Width-based constraints**: Compact sizing with no fixed height limitations

### 🧠 **AI Assistant Functionality**
- **Context awareness**: Analyzes current page content and provides relevant responses
- **Page-specific help**: Understands what page the user is viewing and offers appropriate assistance
- **Multiple AI models**: Uses the existing Grok AI model with potential for model selection
- **Conversation memory**: Maintains chat history during the session
- **Error handling**: Graceful fallbacks for API issues

### 🔧 **Integration Features**
- **Global availability**: Works across all authenticated pages where pinned avatar is visible
- **Maintains 3D capabilities**: Preserves existing avatar manipulation and positioning
- **State persistence**: Chat state survives page navigation and browser sessions
- **Authentication aware**: Uses existing user authentication system

## Technical Implementation

### Components Created

1. **`AvatarChatInterface.jsx`** - Main chat interface component
2. **`usePageContext.js`** - Hook for detecting page content and context
3. **`avatarChatApiSlice.js`** - API integration for avatar-specific chat

### Components Modified

1. **`GlobalPinnedAvatar.jsx`** - Added double-click handler and chat integration
2. **`avatarSlice.js`** - Extended state management for chat functionality
3. **`useAvatarPersistence.js`** - Added chat state persistence

### State Management

New Redux state added to `avatarSlice`:
```javascript
{
  isChatOpen: false,        // Chat interface visibility
  chatHistory: [],          // Message history
  chatContext: null,        // Current page context
  isTyping: false,         // AI typing indicator
}
```

### API Integration

- Extends existing chat API with context-aware messaging
- Sends page context along with user messages for better AI responses
- Maintains separate chat history from main chatbot
- Uses existing authentication and user ID system

## Usage Instructions

### For Users

1. **Enable Pin Mode**: Go to Avatar Selection → Pin Mode tab
2. **Position Avatar**: Drag the floating avatar to desired location
3. **Start Chatting**: Double-click the avatar to open chat interface
4. **Ask Questions**: Type questions about the current page or general topics
5. **Minimize/Close**: Use header controls to minimize or close chat

### For Developers

#### Adding New Page Context

To add context awareness for new pages, update `usePageContext.js`:

```javascript
// Add route context in getRouteContext()
'/new-page': {
  type: 'custom',
  description: 'User is on the new page',
  features: ['feature1', 'feature2'],
},

// Add page-specific data extraction in extractPageSpecificData()
case '/new-page':
  const pageData = extractCustomData();
  return { customData: pageData };
```

#### Customizing AI Responses

Modify `avatarChatApiSlice.js` to customize system prompts:

```javascript
// Update createAvatarSystemPrompt() function
const contextualPrompts = {
  custom: "Help with custom page functionality",
  // ... other prompts
};
```

## Design Philosophy

### Visual Consistency
- **Glass morphism styling**: Matches existing Gurukul platform components
- **Color scheme**: Blue/white/transparent palette consistent with platform
- **Typography**: Monospace font for terminal feel with platform-appropriate colors
- **Animations**: Smooth transitions and typewriter effects
- **Positioning**: Smart boundary detection with platform-style glow effects

### Terminal-Inspired Functionality
- **Command-line prompts**: `>` for user input, `$` for AI responses
- **Typewriter animations**: Character-by-character typing for authentic feel
- **Minimal controls**: No header buttons, click-outside dismissal
- **Context awareness**: Intelligent responses based on current page

## Configuration

### Default Settings

- **AI Model**: Grok (configurable)
- **Chat Position**: Right of avatar (auto-adjusts for screen boundaries)
- **Chat Size**: 280px width, content-driven height
- **Persistence**: Enabled across sessions
- **Context Extraction**: Automatic page content analysis

### Customization Options

- Chat interface styling via CSS classes
- AI model selection (extend existing model picker)
- Custom context extraction rules
- Position preferences and constraints

## Troubleshooting

### Common Issues

1. **Chat not opening**: Ensure pin mode is enabled and avatar is visible
2. **No AI responses**: Check network connection and API availability
3. **Position issues**: Chat auto-adjusts for screen boundaries
4. **Context not working**: Verify page content is properly structured

### Debug Information

- Chat state is logged in Redux DevTools
- API calls are logged in browser console
- Page context extraction can be monitored in `usePageContext` hook

## Future Enhancements

- Voice input/output capabilities
- Custom avatar personalities
- Advanced context understanding
- Integration with learning progress
- Multi-language support
- Conversation export/import

## Dependencies

- React 18+
- Redux Toolkit
- React Router
- Existing chat API infrastructure
- Three.js (for 3D avatar rendering)
- Lucide React (for icons)
