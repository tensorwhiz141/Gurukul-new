import React from "react";
import GlassContainer from "../components/GlassContainer";
import { useQuery } from "@tanstack/react-query";
import CenteredLoader from "../components/CenteredLoader";
import { getTests } from "../api";
import { ExternalLink } from "lucide-react";

export default function Test() {
  const {
    data: tests,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ["tests"],
    queryFn: getTests,
  });

  return (
    <GlassContainer>
      <h2
        className="text-4xl md:text-5xl font-extrabold mb-6 drop-shadow-lg transition-all duration-300 hover:bg-gradient-to-r hover:from-white hover:to-[#FF9933] hover:bg-clip-text hover:text-transparent"
        style={{ color: "#FFFFFF", fontFamily: "Nunito, sans-serif" }}
      >
        Test
      </h2>
      <p
        className="text-lg md:text-xl font-medium mb-4"
        style={{ color: "#FFFFFF", fontFamily: "Nunito, sans-serif" }}
      >
        Take tests to evaluate your knowledge here.
      </p>
      {isLoading ? (
        <div className="relative" style={{ height: "calc(100vh - 350px)" }}>
          <CenteredLoader />
        </div>
      ) : isError ? (
        <p className="text-red-500">
          {error?.message || "Failed to fetch tests."}
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-8">
          {tests &&
            tests.map((test) => (
              <a
                key={test.test_link}
                href={test.test_link}
                target="_blank"
                rel="noopener noreferrer"
                className="flip-card w-72 h-96 mx-auto group focus:outline-none focus:ring-4 focus:ring-orange-300 transition-shadow hover:shadow-2xl"
                title={`Go to test: ${test.title}`}
              >
                <div className="flip-card-inner">
                  <div className="flip-card-front flex flex-col items-center justify-center bg-gradient-to-br from-orange-100/20 to-yellow-100/10 rounded-xl shadow-lg border border-orange-300/30 group-hover:scale-105 transition-transform">
                    <img
                      src={test.image}
                      alt={test.title}
                      className="w-44 h-44 object-cover rounded-md mb-2 border-4 border-white/30 shadow-lg"
                    />
                    <h3 className="text-lg font-bold text-white drop-shadow-lg text-center px-2 mt-2 group-hover:text-orange-300 transition-colors">
                      {test.title}
                    </h3>
                  </div>
                  <div className="flip-card-back flex flex-col items-center justify-center bg-gradient-to-br from-orange-400 to-yellow-300 rounded-xl shadow-lg border border-white/20 p-4 group-hover:scale-105 transition-transform">
                    <h3 className="text-lg font-bold text-white drop-shadow-lg text-center mb-2">
                      {test.title}
                    </h3>
                    <p className="text-white text-sm text-center mb-4 line-clamp-4">
                      {test.description}
                    </p>
                    <span
                      className="inline-flex items-center gap-2 px-5 py-2 bg-white text-orange-500 font-bold rounded-lg shadow-lg hover:bg-orange-500 hover:text-white transition-colors text-base mt-2 border-2 border-orange-400 group-hover:scale-110 group-hover:shadow-xl"
                      style={{ boxShadow: "0 4px 20px rgba(255,153,51,0.15)" }}
                    >
                      Take Test <ExternalLink className="w-4 h-4 ml-1" />
                    </span>
                  </div>
                </div>
              </a>
            ))}
        </div>
      )}
      {/* Flip card styles */}
      <style>{`
        .flip-card {
          perspective: 1000px;
        }
        .flip-card-inner {
          position: relative;
          width: 100%;
          height: 100%;
          transition: transform 0.6s cubic-bezier(.4,2,.6,1);
          transform-style: preserve-3d;
        }
        .flip-card:hover .flip-card-inner {
          transform: rotateY(180deg);
        }
        .flip-card-front, .flip-card-back {
          position: absolute;
          width: 100%;
          height: 100%;
          backface-visibility: hidden;
          border-radius: 0.75rem;
        }
        .flip-card-front {
          z-index: 2;
        }
        .flip-card-back {
          transform: rotateY(180deg);
          z-index: 1;
          display: flex;
          align-items: center;
          justify-content: center;
        }
      `}</style>
    </GlassContainer>
  );
}
