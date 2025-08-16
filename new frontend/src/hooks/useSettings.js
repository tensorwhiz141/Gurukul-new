import { useSelector, useDispatch } from "react-redux";
import { useCallback } from "react";
import {
  selectTheme,
  selectFontSize,
  selectLanguage,
  selectNotifications,
  selectAudioEnabled,
  selectAudioVolume,
  selectAllSettings,
  setTheme,
  setFontSize,
  setLanguage,
  setNotifications,
  setAudioEnabled,
  setAudioVolume,
  updateSettings,
} from "../store/settingsSlice";

/**
 * Custom hook for application settings
 * Provides access to settings state and methods
 */
export const useSettings = () => {
  const dispatch = useDispatch();
  const theme = useSelector(selectTheme);
  const fontSize = useSelector(selectFontSize);
  const language = useSelector(selectLanguage);
  const notifications = useSelector(selectNotifications);
  const audioEnabled = useSelector(selectAudioEnabled);
  const audioVolume = useSelector(selectAudioVolume);
  const allSettings = useSelector(selectAllSettings);

  /**
   * Update the theme
   */
  const updateTheme = useCallback(
    (newTheme) => {
      dispatch(setTheme(newTheme));
    },
    [dispatch]
  );

  /**
   * Update the font size
   */
  const updateFontSize = useCallback(
    (newFontSize) => {
      dispatch(setFontSize(newFontSize));
    },
    [dispatch]
  );

  /**
   * Update the language
   */
  const updateLanguage = useCallback(
    (newLanguage) => {
      dispatch(setLanguage(newLanguage));
    },
    [dispatch]
  );

  /**
   * Update notifications setting
   */
  const updateNotifications = useCallback(
    (enabled) => {
      dispatch(setNotifications(enabled));
    },
    [dispatch]
  );

  /**
   * Update audio enabled setting
   */
  const updateAudioEnabled = useCallback(
    (enabled) => {
      dispatch(setAudioEnabled(enabled));
    },
    [dispatch]
  );

  /**
   * Update audio volume setting
   */
  const updateAudioVolume = useCallback(
    (volume) => {
      dispatch(setAudioVolume(volume));
    },
    [dispatch]
  );

  /**
   * Update multiple settings at once
   */
  const updateMultipleSettings = useCallback(
    (settings) => {
      dispatch(updateSettings(settings));
    },
    [dispatch]
  );

  /**
   * Toggle theme between light and dark
   */
  const toggleTheme = useCallback(() => {
    const newTheme = theme === "dark" ? "light" : "dark";
    dispatch(setTheme(newTheme));
  }, [theme, dispatch]);

  /**
   * Toggle notifications
   */
  const toggleNotifications = useCallback(() => {
    dispatch(setNotifications(!notifications));
  }, [notifications, dispatch]);

  /**
   * Toggle audio
   */
  const toggleAudio = useCallback(() => {
    dispatch(setAudioEnabled(!audioEnabled));
  }, [audioEnabled, dispatch]);

  return {
    theme,
    fontSize,
    language,
    notifications,
    audioEnabled,
    audioVolume,
    allSettings,
    isDarkMode: theme === "dark",
    updateTheme,
    updateFontSize,
    updateLanguage,
    updateNotifications,
    updateAudioEnabled,
    updateAudioVolume,
    updateMultipleSettings,
    toggleTheme,
    toggleNotifications,
    toggleAudio,
  };
};
