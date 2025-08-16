import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

/**
 * Hook to detect current page context and extract relevant information
 * for the avatar chatbot to provide contextually aware responses
 */
export const usePageContext = () => {
  const location = useLocation();
  const [pageContext, setPageContext] = useState(null);

  useEffect(() => {
    const extractPageContext = () => {
      const pathname = location.pathname;
      const pageTitle = document.title;
      
      // Extract main content from the page
      const mainContent = document.querySelector('main') || document.querySelector('[role="main"]') || document.body;
      const headings = Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent?.trim()).filter(Boolean);
      const pageText = mainContent?.textContent?.trim() || '';
      
      // Get route-specific context
      const routeContext = getRouteContext(pathname);
      
      // Extract key information (limit text to avoid overwhelming the AI)
      const contextText = pageText.length > 1000 ? pageText.substring(0, 1000) + '...' : pageText;
      
      const context = {
        pathname,
        pageTitle,
        routeContext,
        headings: headings.slice(0, 5), // Limit to first 5 headings
        contentSummary: contextText,
        timestamp: new Date().toISOString(),
        // Add specific page data based on route
        pageData: extractPageSpecificData(pathname),
      };

      setPageContext(context);
    };

    // Extract context immediately
    extractPageContext();

    // Re-extract context when DOM changes (debounced)
    let timeoutId;
    const observer = new MutationObserver(() => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(extractPageContext, 500);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: false,
    });

    return () => {
      observer.disconnect();
      clearTimeout(timeoutId);
    };
  }, [location.pathname]);

  return pageContext;
};

/**
 * Get context information specific to the current route
 */
const getRouteContext = (pathname) => {
  const routeMap = {
    '/home': {
      type: 'dashboard',
      description: 'User is on the home/dashboard page',
      features: ['overview', 'navigation', 'quick access'],
    },
    '/dashboard': {
      type: 'dashboard',
      description: 'User is viewing the main dashboard',
      features: ['analytics', 'overview', 'metrics'],
    },
    '/subjects': {
      type: 'academic',
      description: 'User is browsing available subjects',
      features: ['subject selection', 'course catalog', 'learning paths'],
    },
    '/learn': {
      type: 'learning',
      description: 'User is in the learning/summarizer section',
      features: ['document upload', 'AI summarization', 'content analysis'],
    },
    '/learn/summary': {
      type: 'learning',
      description: 'User is viewing a generated summary',
      features: ['summary content', 'document analysis', 'key insights'],
    },
    '/chatbot': {
      type: 'ai_interaction',
      description: 'User is in the main chatbot interface',
      features: ['AI conversation', 'multiple models', 'chat history'],
    },
    '/test': {
      type: 'assessment',
      description: 'User is taking or viewing tests',
      features: ['assessments', 'quizzes', 'evaluation'],
    },
    '/lectures': {
      type: 'learning',
      description: 'User is browsing lectures',
      features: ['video content', 'educational materials', 'course content'],
    },
    '/agent-simulator': {
      type: 'ai_interaction',
      description: 'User is in the agent simulator',
      features: ['AI agents', 'simulation', 'interactive learning'],
    },
    '/avatar-selection': {
      type: 'customization',
      description: 'User is selecting or customizing their avatar',
      features: ['3D avatars', 'personalization', 'avatar settings'],
    },
    '/about': {
      type: 'information',
      description: 'User is viewing information about the platform',
      features: ['platform info', 'features', 'documentation'],
    },
    '/settings': {
      type: 'configuration',
      description: 'User is in the settings page',
      features: ['preferences', 'account settings', 'configuration'],
    },
  };

  return routeMap[pathname] || {
    type: 'unknown',
    description: `User is on ${pathname}`,
    features: [],
  };
};

/**
 * Extract page-specific data based on the current route
 */
const extractPageSpecificData = (pathname) => {
  try {
    switch (pathname) {
      case '/subjects':
        // Extract subject information
        const subjects = Array.from(document.querySelectorAll('[data-subject], .subject-card')).map(el => ({
          name: el.textContent?.trim(),
          id: el.dataset?.subject || el.id,
        })).filter(s => s.name);
        return { subjects: subjects.slice(0, 10) };

      case '/learn':
      case '/learn/summary':
        // Extract learning content information
        const uploadedFile = document.querySelector('input[type="file"]')?.files?.[0]?.name;
        const selectedModel = localStorage.getItem('selectedAIModel');
        return { uploadedFile, selectedModel };

      case '/chatbot':
        // Extract chat context
        const messages = Array.from(document.querySelectorAll('.message, [data-message]')).length;
        const currentModel = localStorage.getItem('selectedAIModel');
        return { messageCount: messages, currentModel };

      case '/avatar-selection':
        // Extract avatar context
        const selectedAvatar = localStorage.getItem('gurukul_selected_avatar');
        const favorites = JSON.parse(localStorage.getItem('gurukul_favorite_avatars') || '[]');
        return { selectedAvatar, favoriteCount: favorites.length };

      default:
        return {};
    }
  } catch (error) {
    console.warn('Error extracting page-specific data:', error);
    return {};
  }
};

/**
 * Format page context for AI consumption
 */
export const formatContextForAI = (pageContext) => {
  if (!pageContext) return '';

  const { pathname, pageTitle, routeContext, headings, contentSummary, pageData } = pageContext;

  return `
Current Page Context:
- Page: ${pageTitle} (${pathname})
- Type: ${routeContext.type}
- Description: ${routeContext.description}
- Available Features: ${routeContext.features.join(', ')}
- Main Headings: ${headings.join(', ')}
- Page Data: ${JSON.stringify(pageData)}
- Content Preview: ${contentSummary.substring(0, 300)}...

Please provide contextually relevant assistance based on what the user is currently viewing.
  `.trim();
};
