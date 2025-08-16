import React from "react";

export default function NotFound() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-white">
      <section className="w-full max-w-2xl mx-auto rounded-3xl shadow-xl bg-white px-8 py-16 md:py-24 flex flex-col items-center text-center border border-purple-200">
        <h1 className="text-6xl font-extrabold text-purple-700 mb-4 drop-shadow-lg">
          404
        </h1>
        <p className="text-xl text-black font-medium mb-4">Page not found.</p>
      </section>
    </main>
  );
}
