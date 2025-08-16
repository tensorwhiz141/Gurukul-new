import { createSlice } from "@reduxjs/toolkit";

// Get initial language from localStorage or default to English
const getInitialLanguage = () => {
  try {
    const settings = localStorage.getItem("gurukul_settings");
    if (settings) {
      const parsed = JSON.parse(settings);
      return parsed.language || "english";
    }
    return "english";
  } catch (error) {
    console.error("Error reading language from storage:", error);
    return "english";
  }
};

// All translations in one place
export const translations = {
  hindi: {
    Gurukul: "गुरुकुल",
    "Sign Out": "साइन आउट",
    "A modern rebirth of ancient Indian wisdom, brought to life through AI, storytelling, and immersive technology.":
      "AI, कहानी और इमर्सिव तकनीक के माध्यम से जीवन में लाए गए प्राचीन भारतीय ज्ञान का आधुनिक पुनर्जन्म।",
    You: "आप",
    "Guru AI": "गुरु एआई",
    Appearance: "प्रदर्शन",
    // Settings page
    Settings: "सेटिंग्स",
    "Customize your Gurukul experience by adjusting these settings.":
      "इन सेटिंग्स को समायोजित करके अपने गुरुकुल अनुभव को अनुकूलित करें।",
    Display: "प्रदर्शन",
    "Font Size (Browser Zoom)": "फ़ॉन्ट आकार (ब्राउज़र ज़ूम)",
    Small: "छोटा",
    Medium: "मध्यम",
    Large: "बड़ा",
    Preferences: "प्राथमिकताएँ",
    Language: "भाषा",
    English: "अंग्रेज़ी",
    Hindi: "हिन्दी",
    "Enable Notifications": "सूचनाएं सक्षम करें",
    Account: "खाता",
    "Change Password": "पासवर्ड बदलें",
    "Delete Account": "खाता हटाएं",
    "Save Changes": "परिवर्तन सहेजें",
    "Settings saved successfully!": "सेटिंग्स सफलतापूर्वक सहेजी गईं!",
    "English language selected": "अंग्रेज़ी भाषा चुनी गई",
    "Hindi language selected": "हिंदी भाषा चुनी गई",
    "Zoomed in": "ज़ूम इन किया गया",
    "Zoomed out": "ज़ूम आउट किया गया",
    "Zoom reset": "ज़ूम रीसेट किया गया",
    "Password reset email sent! Please check your inbox.":
      "पासवर्ड रीसेट ईमेल भेज दी गई है! कृपया अपना इनबॉक्स देखें।",
    "Failed to send password reset email. Please try again.":
      "पासवर्ड रीसेट ईमेल भेजने में विफल। कृपया पुनः प्रयास करें।",
    "Confirm Account Deletion": "खाता हटाने की पुष्टि करें",
    "This action is permanent and cannot be undone. All your data will be permanently erased.":
      "यह क्रिया स्थायी है और इसे वापस नहीं किया जा सकता है। आपका सभी डेटा स्थायी रूप से मिटा दिया जाएगा।",
    "Please enter your password to confirm:":
      "पुष्टि करने के लिए कृपया अपना पासवर्ड दर्ज करें:",
    "Enter your password": "अपना पासवर्ड दर्ज करें",
    Cancel: "रद्द करें",
    "Confirm Deletion": "हटाने की पुष्टि करें",
    "Please enter your password to confirm account deletion":
      "खाता हटाने की पुष्टि के लिए कृपया अपना पासवर्ड दर्ज करें",
    "Incorrect password. Please try again.":
      "गलत पासवर्ड। कृपया पुनः प्रयास करें।",
    "Account deleted. You will be signed out.":
      "खाता हटा दिया गया है। आप साइन आउट हो जाएंगे।",
    "Error deleting account. Please try again.":
      "खाता हटाने में त्रुटि। कृपया पुनः प्रयास करें।",
    "An unexpected error occurred. Please try again.":
      "एक अनपेक्षित त्रुटि हुई। कृपया पुनः प्रयास करें।",

    // Navigation
    Home: "होम",
    Dashboard: "डैशबोर्ड",
    Subjects: "विषय",
    Summarizer: "सीखें",
    Chatbot: "चैटबॉट",
    About: "हमारे बारे में",
    "3D Models": "3डी मॉडल",

    // 3D Model Viewer
    "3D Cultural Models": "3डी सांस्कृतिक मॉडल",
    "3D Model Viewer Coming Soon": "3डी मॉडल व्यूअर जल्द आ रहा है",
    "We're currently working on implementing our 3D cultural models viewer. This feature will allow you to explore traditional architecture and cultural artifacts in interactive 3D space.":
      "हम वर्तमान में अपने 3डी सांस्कृतिक मॉडल व्यूअर को लागू करने पर काम कर रहे हैं। यह सुविधा आपको इंटरैक्टिव 3डी स्पेस में पारंपरिक वास्तुकला और सांस्कृतिक कलाकृतियों का पता लगाने की अनुमति देगी।",
    "Get Notified When Available": "जब उपलब्ध हो तब सूचित हों",

    // Common UI elements
    "Sign In": "साइन इन",
    "Sign Up": "साइन अप",
    Search: "खोजें",
    Submit: "जमा करें",
    Loading: "लोड हो रहा है",
    Welcome: "स्वागत है",
    Profile: "प्रोफ़ाइल",
    Notifications: "सूचनाएँ",
    Messages: "संदेश",
    Help: "सहायता",
    Error: "त्रुटि",
    Success: "सफलता",

    // Dashboard page
    "User Dashboard": "उपयोगकर्ता डैशबोर्ड",
    "Welcome back! Your progress and tokens will appear here soon.":
      "वापस आने पर स्वागत है! आपकी प्रगति और टोकन जल्द ही यहां दिखाई देंगे।",
    "Your Progress": "आपकी प्रगति",
    "Recent Activity": "हाल की गतिविधि",
    "Completed Lessons": "पूर्ण पाठ",
    "Ongoing Courses": "चालू पाठ्यक्रम",
    "Achievement Points": "उपलब्धि अंक",
    "Knowledge Tokens": "ज्ञान टोकन",
    "View All": "सभी देखें",
    "Continue Learning": "सीखना जारी रखें",
    "Start New Course": "नया पाठ्यक्रम शुरू करें",
    "Recommended for You": "आपके लिए अनुशंसित",

    // Subject page
    "Explore Subjects": "विषयों का अन्वेषण करें",
    "Ancient Texts": "प्राचीन ग्रंथ",
    "Vedic Mathematics": "वैदिक गणित",
    Sanskrit: "संस्कृत",
    "Yoga & Meditation": "योग और ध्यान",
    Ayurveda: "आयुर्वेद",
    "Traditional Arts": "पारंपरिक कला",
    Philosophy: "दर्शनशास्त्र",
    "View Subject": "विषय देखें",

    // Chatbot page
    "Ask Guru": "गुरु से पूछें",
    "Type your question here...": "अपना प्रश्न यहां टाइप करें...",
    Send: "भेजें",
    "AI is thinking...": "AI सोच रहा है...",
    "Try asking about Vedas, Yoga, Ayurveda, or ancient Indian wisdom.":
      "वेद, योग, आयुर्वेद, या प्राचीन भारतीय ज्ञान के बारे में पूछने का प्रयास करें।",
    "Powered by GPT": "GPT द्वारा संचालित",
    "Chat History": "चैट इतिहास",
    "Clear Chat": "चैट साफ करें",
    "Welcome to Gurukul": "गुरुकुल में आपका स्वागत है",
    "Get Started": "शुरू करें",
    "Namaste!🙏 I am UniGuru, your AI learning assistant. How can I help you learn today?":
      "नमस्ते! मैं गुरु हूँ, आपका AI लर्निंग सहायक। मैं आज आपकी कैसे मदद कर सकता हूँ?",
    "Ask about ancient Indian wisdom...":
      "प्राचीन भारतीय ज्ञान के बारे में पूछें...",
    "This AI assistant is designed to help you learn about ancient Indian wisdom and philosophy.":
      "यह AI सहायक आपको प्राचीन भारतीय ज्ञान और दर्शन सीखने में मदद करने के लिए डिज़ाइन किया गया है।",
    "I apologize, but I encountered an error. Please try again later.":
      "माफ़ी चाहता हूँ, लेकिन मुझे एक त्रुटि का सामना करना पड़ा। कृपया बाद में पुनः प्रयास करें।",
  },
};

export const languageSlice = createSlice({
  name: "language",
  initialState: {
    current: getInitialLanguage(),
  },
  reducers: {
    setLanguage: (state, action) => {
      state.current = action.payload;
      try {
        // Also update the settings in localStorage
        const settings = localStorage.getItem("gurukul_settings");
        if (settings) {
          const parsed = JSON.parse(settings);
          parsed.language = action.payload;
          localStorage.setItem("gurukul_settings", JSON.stringify(parsed));
          // Backup to sessionStorage
          sessionStorage.setItem("gurukul_settings", JSON.stringify(parsed));
        } else {
          localStorage.setItem(
            "gurukul_settings",
            JSON.stringify({
              language: action.payload,
              notifications: true,
            })
          );
          // Backup to sessionStorage
          sessionStorage.setItem(
            "gurukul_settings",
            JSON.stringify({
              language: action.payload,
              notifications: true,
            })
          );
        }
        // Update HTML attribute
        document.documentElement.setAttribute("data-language", action.payload);
      } catch (error) {
        console.error("Error updating language in storage:", error);
      }
    },
  },
});

export const { setLanguage } = languageSlice.actions;

// Selector to get current language
export const selectLanguage = (state) => state.language.current;

// Translation function
export const translate = (text, language = null) => {
  // If no language specified, try to get from localStorage
  if (!language) {
    try {
      const settings = localStorage.getItem("gurukul_settings");
      if (settings) {
        const parsed = JSON.parse(settings);
        language = parsed.language;
      }
    } catch (error) {
      console.error("Error reading language from storage:", error);
      language = "english";
    }
  }

  if (language === "hindi" && translations.hindi[text]) {
    return translations.hindi[text];
  }
  return text;
};

export default languageSlice.reducer;
