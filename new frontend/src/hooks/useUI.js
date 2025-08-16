import { useSelector, useDispatch } from 'react-redux';
import { useCallback } from 'react';
import {
  selectSidebarCollapsed,
  selectIsLoading,
  selectActiveModal,
  selectToasts,
  toggleSidebar,
  setSidebarCollapsed,
  setLoading,
  setActiveModal,
  addToast,
  removeToast,
  clearToasts
} from '../store/uiSlice';

/**
 * Custom hook for UI state
 * Provides access to UI state and methods
 */
export const useUI = () => {
  const dispatch = useDispatch();
  const sidebarCollapsed = useSelector(selectSidebarCollapsed);
  const isLoading = useSelector(selectIsLoading);
  const activeModal = useSelector(selectActiveModal);
  const toasts = useSelector(selectToasts);
  
  /**
   * Toggle sidebar collapsed state
   */
  const handleToggleSidebar = useCallback(() => {
    dispatch(toggleSidebar());
  }, [dispatch]);
  
  /**
   * Set sidebar collapsed state
   */
  const handleSetSidebarCollapsed = useCallback((collapsed) => {
    dispatch(setSidebarCollapsed(collapsed));
  }, [dispatch]);
  
  /**
   * Set loading state
   */
  const handleSetLoading = useCallback((loading) => {
    dispatch(setLoading(loading));
  }, [dispatch]);
  
  /**
   * Set active modal
   */
  const handleSetActiveModal = useCallback((modal) => {
    dispatch(setActiveModal(modal));
  }, [dispatch]);
  
  /**
   * Add a toast notification
   */
  const handleAddToast = useCallback((toast) => {
    dispatch(addToast(toast));
    
    // Automatically remove the toast after its duration
    if (toast.duration !== Infinity) {
      setTimeout(() => {
        dispatch(removeToast(toast.id));
      }, toast.duration || 3000);
    }
  }, [dispatch]);
  
  /**
   * Remove a toast notification
   */
  const handleRemoveToast = useCallback((id) => {
    dispatch(removeToast(id));
  }, [dispatch]);
  
  /**
   * Clear all toast notifications
   */
  const handleClearToasts = useCallback(() => {
    dispatch(clearToasts());
  }, [dispatch]);
  
  /**
   * Show a success toast
   */
  const showSuccessToast = useCallback((message, options = {}) => {
    handleAddToast({
      type: 'success',
      message,
      duration: options.duration || 3000,
      ...options
    });
  }, [handleAddToast]);
  
  /**
   * Show an error toast
   */
  const showErrorToast = useCallback((message, options = {}) => {
    handleAddToast({
      type: 'error',
      message,
      duration: options.duration || 5000,
      ...options
    });
  }, [handleAddToast]);
  
  /**
   * Show an info toast
   */
  const showInfoToast = useCallback((message, options = {}) => {
    handleAddToast({
      type: 'info',
      message,
      duration: options.duration || 3000,
      ...options
    });
  }, [handleAddToast]);
  
  /**
   * Show a warning toast
   */
  const showWarningToast = useCallback((message, options = {}) => {
    handleAddToast({
      type: 'warning',
      message,
      duration: options.duration || 4000,
      ...options
    });
  }, [handleAddToast]);
  
  return {
    sidebarCollapsed,
    isLoading,
    activeModal,
    toasts,
    toggleSidebar: handleToggleSidebar,
    setSidebarCollapsed: handleSetSidebarCollapsed,
    setLoading: handleSetLoading,
    setActiveModal: handleSetActiveModal,
    addToast: handleAddToast,
    removeToast: handleRemoveToast,
    clearToasts: handleClearToasts,
    showSuccessToast,
    showErrorToast,
    showInfoToast,
    showWarningToast,
  };
};
