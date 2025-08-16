import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
} from "react";
import { toast } from "react-hot-toast";

const LoaderContext = createContext({
  show: false,
  showLoader: () => {},
  hideLoader: () => {},
  showToastLoader: () => {},
  hideToastLoader: () => {},
});

export function LoaderProvider({ children }) {
  const [show, setShow] = useState(false);
  const toastIdRef = useRef(null);

  // Regular loader functions
  const showLoader = useCallback(() => setShow(true), []);
  const hideLoader = useCallback(() => setShow(false), []);

  // Toast-based loader functions
  const showToastLoader = useCallback((message = "Loading...") => {
    // If there's already a toast loader, dismiss it first
    if (toastIdRef.current) {
      toast.dismiss(toastIdRef.current);
    }

    // Create a new toast loader and store its ID
    toastIdRef.current = toast.loading(message, {
      position: "top-right",
    });

    // Also set the regular loader state
    setShow(true);

    // Return the toast ID for manual dismissal if needed
    return toastIdRef.current;
  }, []);

  const hideToastLoader = useCallback(() => {
    // Dismiss the toast if it exists
    if (toastIdRef.current) {
      toast.dismiss(toastIdRef.current);
      toastIdRef.current = null;
    }

    // Also hide the regular loader
    setShow(false);
  }, []);

  return (
    <LoaderContext.Provider
      value={{
        show,
        showLoader,
        hideLoader,
        showToastLoader,
        hideToastLoader,
      }}
    >
      {children}
    </LoaderContext.Provider>
  );
}

export function useLoader() {
  return useContext(LoaderContext);
}
