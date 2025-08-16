import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

// Create a separate API slice for external APIs
export const externalApiSlice = createApi({
  reducerPath: "externalApi",
  baseQuery: fetchBaseQuery({
    timeout: 5000, // 5 second timeout for external APIs
  }),
  tagTypes: ["Quote", "Fact", "Weather", "Location"],
  endpoints: (builder) => ({
    // Get daily motivational quote
    getDailyQuote: builder.query({
      query: () => ({
        url: "https://api.quotable.io/random?tags=motivational|inspirational",
        method: "GET",
      }),
      transformResponse: (response) => {
        return { q: response.content || "Stay motivated and keep learning!" };
      },
      transformErrorResponse: (response) => {
        // Fallback quotes for when API fails
        const LOCAL_QUOTES = [
          "The only way to do great work is to love what you do.",
          "Innovation distinguishes between a leader and a follower.",
          "Stay hungry, stay foolish.",
          "The future belongs to those who believe in the beauty of their dreams.",
          "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        ];
        const randomQuote = LOCAL_QUOTES[Math.floor(Math.random() * LOCAL_QUOTES.length)];
        return { q: randomQuote };
      },
      providesTags: ["Quote"],
    }),

    // Get random fact
    getRandomFact: builder.query({
      query: () => ({
        url: "http://numbersapi.com/random/trivia?json",
        method: "GET",
      }),
      transformResponse: (response) => {
        return response.text || "Numbers are fascinating!";
      },
      transformErrorResponse: () => {
        const staticFacts = [
          "The human brain contains approximately 86 billion neurons.",
          "A group of flamingos is called a 'flamboyance'.",
          "Honey never spoils - archaeologists have found edible honey in ancient Egyptian tombs.",
          "The shortest war in history lasted only 38-45 minutes.",
          "A single cloud can weigh more than a million pounds.",
        ];
        return staticFacts[Math.floor(Math.random() * staticFacts.length)];
      },
      providesTags: ["Fact"],
    }),

    // Get location by coordinates
    getLocationByCoords: builder.query({
      query: ({ lat, lon }) => ({
        url: `https://api.opencagedata.com/geocode/v1/json?q=${lat}+${lon}&key=16fa592c443b4ee4a8495415749e4c76`,
        method: "GET",
      }),
      providesTags: (result, error, { lat, lon }) => [
        { type: "Location", id: `${lat}-${lon}` },
      ],
    }),

    // Get weather by coordinates
    getWeatherByCoords: builder.query({
      query: ({ lat, lon }) => ({
        url: `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=40c58e845fb8e4636bca94fa940db50e&units=metric`,
        method: "GET",
      }),
      providesTags: (result, error, { lat, lon }) => [
        { type: "Weather", id: `${lat}-${lon}` },
      ],
    }),

    // Combined location and weather query
    getLocationWeather: builder.query({
      queryFn: async ({ lat, lon }, api, extraOptions, baseQuery) => {
        try {
          // Fetch location
          const locationResult = await baseQuery({
            url: `https://api.opencagedata.com/geocode/v1/json?q=${lat}+${lon}&key=16fa592c443b4ee4a8495415749e4c76`,
          });

          // Fetch weather
          const weatherResult = await baseQuery({
            url: `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=40c58e845fb8e4636bca94fa940db50e&units=metric`,
          });

          if (locationResult.error || weatherResult.error) {
            return { error: "Failed to fetch location or weather data" };
          }

          const locationData = locationResult.data;
          const weatherData = weatherResult.data;

          // Extract location info
          const locationInfo = locationData.results?.[0];
          const city = locationInfo?.components?.city || 
                      locationInfo?.components?.town || 
                      locationInfo?.components?.village || 
                      "Unknown";
          const country = locationInfo?.components?.country || "Unknown";

          return {
            data: {
              location: `${city}, ${country}`,
              weather: {
                temperature: Math.round(weatherData.main?.temp || 0),
                description: weatherData.weather?.[0]?.description || "Unknown",
                icon: weatherData.weather?.[0]?.icon || "01d",
              },
            },
          };
        } catch (error) {
          return { error: error.message };
        }
      },
      providesTags: (result, error, { lat, lon }) => [
        { type: "Location", id: `${lat}-${lon}` },
        { type: "Weather", id: `${lat}-${lon}` },
      ],
    }),
  }),
});

export const {
  useGetDailyQuoteQuery,
  useGetRandomFactQuery,
  useGetLocationByCoordsQuery,
  useGetWeatherByCoordsQuery,
  useGetLocationWeatherQuery,
} = externalApiSlice;
