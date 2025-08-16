# Tutorial System Documentation

## 🎓 Overview

The Gurukul Tutorial System provides a comprehensive onboarding experience for new users using floating agents and interactive tooltips. The system guides users through all features of the platform with contextual help and step-by-step instructions.

## ✨ Features

### 🎯 **Intelligent Onboarding**
- **First-time user detection** - Automatically detects new users
- **Progressive disclosure** - Introduces features step-by-step
- **Contextual guidance** - Provides relevant help based on current page
- **Avatar integration** - Uses the existing AI avatar for personalized guidance

### 🏠 **Home Page Enhancements**
- **Interactive tooltips** on hover for all main navigation buttons
- **Detailed feature descriptions** with key benefits
- **Quick navigation** directly from tooltips
- **Visual feedback** with smooth animations

### 📱 **Page-Specific Tutorials**
- **Dashboard** - Progress tracking, achievements, and analytics
- **Subjects** - AI lesson generation and enhanced content
- **Summarizer** - Document upload and analysis features
- **Chatbot** - Chat history and conversation management
- **Tests** - Assessment system and feedback
- **Lectures** - Video content and learning materials

### 🎮 **Interactive Components**
- **Floating tooltips** with rich content and navigation
- **Highlight overlays** that focus attention on specific elements
- **Step-by-step progression** with next/previous controls
- **Skip options** for experienced users

## 🛠️ Technical Implementation

### **State Management**
```javascript
// Redux slice for tutorial state
src/store/tutorialSlice.js
- Tutorial progress tracking
- User preferences
- Step management
- Settings persistence
```

### **Core Components**
```
src/components/tutorial/
├── TutorialManager.jsx      # Main tutorial orchestrator
├── TutorialTooltip.jsx      # Interactive tooltip component
├── TutorialHighlight.jsx    # Element highlighting
├── TutorialOverlay.jsx      # Focus overlay with cutouts
├── HomeButtonTooltip.jsx    # Home page hover tooltips
└── TutorialControls.jsx     # User controls and settings
```

### **Integration Points**
- **Layout.jsx** - Tutorial system initialization
- **Home.jsx** - Enhanced with hover tooltips and tutorial data attributes
- **All main pages** - Tutorial data attributes for targeting
- **AvatarChatInterface.jsx** - Tutorial message handling
- **GlobalPinnedAvatar.jsx** - Tutorial integration

## 🎨 User Experience

### **First-Time User Journey**
1. **Welcome** - Greeting and platform introduction
2. **Navigation** - Overview of main features
3. **Dashboard** - Progress tracking explanation
4. **Subjects** - AI lesson generation demo
5. **Summarizer** - Document analysis walkthrough
6. **Chatbot** - Chat features and history
7. **Tests** - Assessment system overview
8. **Lectures** - Video content introduction
9. **Avatar** - AI companion explanation
10. **Completion** - Tutorial finished, ready to explore

### **Returning User Experience**
- **Smart detection** - Recognizes returning users
- **Optional tutorials** - Can restart tutorials for specific pages
- **Settings control** - Customize tutorial behavior
- **Quick help** - Access tutorials anytime via controls

## 🎛️ User Controls

### **Tutorial Settings**
- **Auto-start tutorials** - Enable/disable automatic tutorial start
- **Skip completed** - Skip tutorials for already completed pages
- **Avatar messages** - Show/hide avatar tutorial messages
- **Feature tooltips** - Enable/disable hover tooltips
- **Reset progress** - Clear all tutorial progress

### **Navigation Controls**
- **Next/Previous** - Step through tutorial at own pace
- **Skip tutorial** - Skip current page tutorial
- **Close** - Exit tutorial system
- **Restart** - Begin tutorial again

## 🔧 Configuration

### **Tutorial Steps**
Each page has configurable tutorial steps defined in `tutorialSlice.js`:

```javascript
const TUTORIAL_STEPS = {
  home: [
    {
      id: 'welcome',
      target: '.home-container',
      title: '🎓 Welcome to Gurukul!',
      content: 'Your AI-powered learning companion...',
      position: 'center',
      showAvatar: true,
      avatarMessage: 'Welcome message...',
    },
    // ... more steps
  ],
  // ... other pages
};
```

### **Tooltip Content**
Home page button tooltips are configured with:

```javascript
const HOME_BUTTON_TOOLTIPS = {
  dashboard: {
    title: '📊 Dashboard',
    description: 'Your learning command center...',
    features: ['Progress tracking', 'Achievement system', ...],
  },
  // ... other buttons
};
```

## 🎯 Data Attributes

Tutorial targeting uses data attributes on key elements:

```html
<!-- Page containers -->
<div data-tutorial="dashboard-container">
<div data-tutorial="subjects-container">

<!-- Interactive elements -->
<button data-tutorial="dashboard-button">
<form data-tutorial="lesson-form">
<div data-tutorial="file-upload">

<!-- Navigation -->
<aside data-tutorial="sidebar">
<div data-tutorial="global-pinned-avatar">
```

## 🚀 Usage

### **For Users**
1. **First visit** - Tutorial starts automatically on home page
2. **Hover tooltips** - Hover over home page buttons for details
3. **Manual start** - Click tutorial button (bottom-right) on any page
4. **Settings** - Click settings gear to customize experience
5. **Avatar help** - Double-click avatar for contextual assistance

### **For Developers**
1. **Add new pages** - Define tutorial steps in `tutorialSlice.js`
2. **Add data attributes** - Use `data-tutorial="element-id"` for targeting
3. **Customize content** - Update step content and avatar messages
4. **Extend components** - Create new tutorial components as needed

## 🎨 Styling

Tutorial styles are defined in `src/styles/tutorial.css`:
- **Animations** - Smooth fade-ins and pulse effects
- **Responsive design** - Mobile-friendly layouts
- **Theme support** - Dark/light mode compatibility
- **Accessibility** - High contrast and keyboard navigation

## 🔮 Future Enhancements

- **Voice guidance** - Audio narration for tutorials
- **Interactive demos** - Hands-on practice sessions
- **Progress analytics** - Track tutorial completion rates
- **Personalization** - Adaptive tutorials based on user behavior
- **Multi-language** - Internationalization support
- **Advanced targeting** - Dynamic element detection

## 🧪 Testing

The tutorial system includes:
- **State persistence** - Progress saved across sessions
- **Error handling** - Graceful fallbacks for missing elements
- **Performance optimization** - Efficient rendering and animations
- **Cross-browser compatibility** - Works across modern browsers

## 📊 Analytics

Tutorial system tracks:
- **Completion rates** - How many users complete tutorials
- **Drop-off points** - Where users exit tutorials
- **Feature adoption** - Which features users explore after tutorials
- **User preferences** - Settings and customization choices

This comprehensive tutorial system ensures that new users can quickly understand and effectively use all features of the Gurukul learning platform.
