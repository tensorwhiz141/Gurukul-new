import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  sidebarCollapsed: false,
  isLoading: false,
  activeModal: null,
  toasts: [],
};

export const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    setSidebarCollapsed: (state, action) => {
      state.sidebarCollapsed = action.payload;
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setActiveModal: (state, action) => {
      state.activeModal = action.payload;
    },
    addToast: (state, action) => {
      state.toasts.push({
        id: Date.now(),
        ...action.payload,
      });
    },
    removeToast: (state, action) => {
      state.toasts = state.toasts.filter(toast => toast.id !== action.payload);
    },
    clearToasts: (state) => {
      state.toasts = [];
    },
  },
});

// Export actions
export const {
  toggleSidebar,
  setSidebarCollapsed,
  setLoading,
  setActiveModal,
  addToast,
  removeToast,
  clearToasts,
} = uiSlice.actions;

// Export selectors
export const selectSidebarCollapsed = (state) => state.ui.sidebarCollapsed;
export const selectIsLoading = (state) => state.ui.isLoading;
export const selectActiveModal = (state) => state.ui.activeModal;
export const selectToasts = (state) => state.ui.toasts;

export default uiSlice.reducer;
