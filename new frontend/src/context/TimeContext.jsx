import React, { useState, useCallback, useEffect, useRef } from "react";
import { supabase } from "../supabaseClient";
import { TimeContext } from "./TimeContextDef";

export const TimeProvider = ({ children }) => {
  const [totalTimeToday, setTotalTimeToday] = useState(0);
  const [currentSessionTime, setCurrentSessionTime] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [timeHistory, setTimeHistory] = useState([]);
  const initRef = useRef(false);

  const fetchTodayTime = useCallback(async (userId) => {
    try {
      if (!userId) return 0;

      const startOfDay = new Date();
      startOfDay.setHours(0, 0, 0, 0);

      const { data, error: fetchError } = await supabase
        .from("time_tracking")
        .select("time_spent")
        .eq("user_id", userId)
        .gte("created_at", startOfDay.toISOString());

      if (fetchError) throw fetchError;

      const total =
        data?.reduce((sum, rec) => sum + (rec.time_spent || 0), 0) || 0;
      setTotalTimeToday(total);
      setIsLoading(false);
      return total;
    } catch (err) {
      console.error("Error fetching time:", err);
      setIsLoading(false);
      return 0;
    }
  }, []);

  // Fetch time history for stats (grouped by day)
  const fetchTimeHistory = useCallback(async (userId) => {
    try {
      if (!userId) return [];
      const { data, error } = await supabase
        .from("time_tracking")
        .select("time_spent, created_at")
        .eq("user_id", userId);
      if (error) throw error;
      // Group by date
      const dayTotals = {};
      data.forEach((rec) => {
        const date = new Date(rec.created_at).toISOString().slice(0, 10);
        dayTotals[date] = (dayTotals[date] || 0) + (rec.time_spent || 0);
      });
      // Convert to array sorted by date descending
      const history = Object.entries(dayTotals)
        .map(([date, total]) => ({ date, total }))
        .sort((a, b) => new Date(b.date) - new Date(a.date));
      setTimeHistory(history);
      return history;
    } catch (err) {
      console.error("Error fetching time history:", err);
      setTimeHistory([]);
      return [];
    }
  }, []);

  const handleTimeUpdate = useCallback((elapsed) => {
    setCurrentSessionTime(elapsed);
    // Once we get the first time update, we know the session is active
    setIsLoading(false);
  }, []);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const initializeTime = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session?.user) {
        await fetchTodayTime(session.user.id);
        await fetchTimeHistory(session.user.id);
      } else {
        setIsLoading(false);
      }
    };

    initializeTime();

    const subscription = supabase.auth.onAuthStateChange(
      async (event, session) => {
        if (event === "SIGNED_IN" && session?.user) {
          await fetchTodayTime(session.user.id);
          await fetchTimeHistory(session.user.id);
        } else if (event === "SIGNED_OUT") {
          setTotalTimeToday(0);
          setCurrentSessionTime(0);
          setTimeHistory([]);
          setIsLoading(false);
        }
      }
    );

    return () => {
      subscription.data?.subscription?.unsubscribe();
    };
  }, [fetchTodayTime, fetchTimeHistory]);

  const value = {
    totalTimeToday,
    currentSessionTime,
    isLoading: isLoading && !currentSessionTime, // Only show loading if we have no time data
    handleTimeUpdate,
    fetchTodayTime,
    timeHistory,
  };

  return <TimeContext.Provider value={value}>{children}</TimeContext.Provider>;
};
