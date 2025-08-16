import React, { createContext, useContext, useState, useEffect } from "react";
import i18n from "../i18n";

// Create settings context
const SettingsContext = createContext({
  theme: "dark",
  fontSize: "medium",
  language: "english",
  notifications: true,
  updateSettings: () => {},
});

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error("useSettings must be used within a ThemeProvider");
  }
  return context;
}

export function ThemeProvider({ children }) {
  // Initialize state from localStorage or use defaults
  const [settings, setSettings] = useState(() => {
    try {
      const savedSettings = localStorage.getItem("gurukul_settings");
      return savedSettings
        ? JSON.parse(savedSettings)
        : {
            theme: "dark",
            fontSize: "medium",
            language: "english",
            notifications: true,
          };
    } catch (error) {
      console.error("Error loading settings:", error);
      return {
        theme: "dark",
        fontSize: "medium",
        language: "english",
        notifications: true,
      };
    }
  });

  // Update settings and save to localStorage
  const updateSettings = (newSettings) => {
    setSettings((prevSettings) => {
      const updatedSettings = { ...prevSettings, ...newSettings };
      try {
        localStorage.setItem(
          "gurukul_settings",
          JSON.stringify(updatedSettings)
        );
      } catch (error) {
        console.error("Error saving settings:", error);
      }
      return updatedSettings;
    });
  };

  // Apply theme to document based on current settings
  useEffect(() => {
    // Apply dark/light mode
    const applyTheme = (themeName) => {
      if (themeName === "light") {
        document.documentElement.classList.remove("dark");
        document.documentElement.style.backgroundColor = "transparent";
      } else if (themeName === "dark") {
        document.documentElement.classList.add("dark");
        document.documentElement.style.backgroundColor = "transparent";
      } else if (themeName === "auto") {
        if (
          window.matchMedia &&
          window.matchMedia("(prefers-color-scheme: dark)").matches
        ) {
          document.documentElement.classList.add("dark");
          document.documentElement.style.backgroundColor = "transparent";
        } else {
          document.documentElement.classList.remove("dark");
          document.documentElement.style.backgroundColor = "transparent";
        }
      }
    };

    // Apply font size
    const applyFontSize = (size) => {
      const fontSizes = {
        small: "14px",
        medium: "16px",
        large: "18px",
      };
      document.documentElement.style.fontSize = fontSizes[size] || "16px";
    };

    applyTheme(settings.theme);
    applyFontSize(settings.fontSize);

    // Listen for system theme changes if using auto theme
    if (settings.theme === "auto") {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      const handleChange = () => applyTheme("auto");
      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    }
  }, [settings.theme, settings.fontSize]);

  // Sync language changes to i18next
  useEffect(() => {
    i18n.changeLanguage(settings.language);
  }, [settings.language]);

  return (
    <SettingsContext.Provider value={{ ...settings, updateSettings }}>
      {children}
    </SettingsContext.Provider>
  );
}
