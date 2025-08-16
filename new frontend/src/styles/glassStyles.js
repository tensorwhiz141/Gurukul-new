/**
 * Glassmorphic UI Style System
 * 
 * This file contains reusable glassmorphic style variables and utility functions
 * to maintain consistent styling throughout the application.
 */

// Core glass effect styles
export const glassEffect = {
  background: 'rgba(255, 255, 255, 0.15)',
  backdropFilter: 'blur(12px)',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.1)',
};

// Glass panel variants
export const glassPanels = {
  // Primary content panels (MainBoard)
  primary: {
    ...glassEffect,
    background: 'rgba(255, 255, 255, 0.18)',
    border: '2px solid rgba(255, 215, 0, 0.3)',
    boxShadow: '0 8px 32px 0 rgba(214, 167, 108, 0.25), 0 1px 8px 0 rgba(255, 215, 0, 0.08) inset',
  },
  
  // Secondary panels (Sidebar, Header)
  secondary: {
    ...glassEffect,
    background: 'rgba(255, 255, 255, 0.12)',
    border: '1px solid rgba(255, 215, 0, 0.2)',
    boxShadow: '0 4px 24px 0 rgba(214, 167, 108, 0.15)',
  },
  
  // Form panels (SignIn, SignUp)
  form: {
    ...glassEffect,
    background: 'rgba(255, 255, 255, 0.2)',
    border: '1px solid rgba(255, 215, 0, 0.25)',
    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
  },
};

// Interactive elements
export const glassInteractive = {
  // Buttons
  button: {
    ...glassEffect,
    background: 'rgba(255, 255, 255, 0.2)',
    border: '1px solid rgba(255, 215, 0, 0.3)',
    color: '#FFF',
    transition: 'all 0.3s ease',
    hover: {
      background: 'rgba(255, 153, 51, 0.25)',
      border: '1px solid rgba(255, 215, 0, 0.5)',
      transform: 'translateY(-2px)',
      boxShadow: '0 10px 40px 0 rgba(31, 38, 135, 0.15)',
    },
    active: {
      transform: 'translateY(1px)',
      boxShadow: '0 5px 15px 0 rgba(31, 38, 135, 0.1)',
    },
  },
  
  // Input fields
  input: {
    ...glassEffect,
    background: 'rgba(255, 255, 255, 0.15)',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    color: '#FFF',
    placeholder: {
      color: 'rgba(255, 255, 255, 0.7)',
    },
    focus: {
      border: '1px solid rgba(255, 215, 0, 0.5)',
      boxShadow: '0 0 0 2px rgba(255, 215, 0, 0.2)',
    },
  },
};

// Typography
export const glassTypography = {
  heading: {
    fontFamily: "'Nunito', sans-serif",
    color: '#5D001E',
  },
  body: {
    fontFamily: "'Nunito', sans-serif",
    color: '#2E4A3F',
  },
  light: {
    color: '#FFF',
  },
};

// Color palette
export const colors = {
  saffron: '#FF9933',
  deepMaroon: '#5D001E',
  sandstone: '#D6A76C',
  forestGreen: '#2E4A3F',
  gold: '#FFD700',
  indigoInk: '#1E1B4B',
  electricBlue: '#00CFFF',
  warmWhite: '#FDF6E3',
};

// Helper function to generate tailwind glass classes
export const getTailwindGlassClasses = (type = 'primary') => {
  switch (type) {
    case 'button':
      return 'bg-white/20 backdrop-blur-lg border border-[#FFD700]/30 shadow-lg transition-all duration-300 hover:bg-[#FF9933]/25 hover:border-[#FFD700]/50 hover:-translate-y-0.5 active:translate-y-0.5';
    case 'input':
      return 'bg-white/15 backdrop-blur-lg border border-white/20 shadow-md focus:border-[#FFD700]/50 focus:shadow-[0_0_0_2px_rgba(255,215,0,0.2)]';
    case 'form':
      return 'bg-white/20 backdrop-blur-xl border border-[#FFD700]/25 shadow-xl';
    case 'secondary':
      return 'bg-white/12 backdrop-blur-lg border border-[#FFD700]/20 shadow-lg';
    case 'primary':
    default:
      return 'bg-white/18 backdrop-blur-xl border-2 border-[#FFD700]/30 shadow-xl shadow-[#D6A76C]/25';
  }
};
