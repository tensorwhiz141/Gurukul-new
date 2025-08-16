import React, { useEffect, useRef } from "react";
import gsap from "gsap";

/**
 * AnimatedList - Reusable component for animated list items
 * Props:
 * - items: array of items to render
 * - renderItem: function to render each item
 * - className: additional classes for the list container
 */
export default function AnimatedList({ items, renderItem, className = "" }) {
  const listRef = useRef(null);
  const itemRefs = useRef(new Map());

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;

    // Initial list entrance animation
    gsap.fromTo(
      list,
      {
        opacity: 0,
        y: 30,
      },
      {
        opacity: 1,
        y: 0,
        duration: 0.6,
        ease: "power2.out",
      }
    );
  }, []);

  useEffect(() => {
    const currentItemRefs = new Map(itemRefs.current);
    const currentItems = items;

    // Find new items
    const newItems = currentItems.filter(
      (item) => !currentItemRefs.has(item.id)
    );

    // Find removed items
    const removedItems = Array.from(currentItemRefs.keys()).filter(
      (id) => !currentItems.find((item) => item.id === id)
    );

    // Animate new items
    newItems.forEach((item) => {
      const element = itemRefs.current.get(item.id);
      if (!element) return;

      gsap.fromTo(
        element,
        {
          opacity: 0,
          x: -20,
          scale: 0.9,
        },
        {
          opacity: 1,
          x: 0,
          scale: 1,
          duration: 0.4,
          ease: "power2.out",
        }
      );
    });

    // Animate removed items
    removedItems.forEach((id) => {
      const element = currentItemRefs.get(id);
      if (!element) return;

      gsap.to(element, {
        opacity: 0,
        x: 20,
        scale: 0.9,
        duration: 0.3,
        ease: "power2.in",
        onComplete: () => {
          currentItemRefs.delete(id);
        },
      });
    });

    // Update hover animations for all current items
    currentItems.forEach((item) => {
      const element = itemRefs.current.get(item.id);
      if (!element) return;

      const hoverTimeline = gsap.timeline({ paused: true }).to(element, {
        scale: 1.02,
        backgroundColor: "rgba(255, 255, 255, 0.08)",
        boxShadow:
          "0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(255, 215, 0, 0.1) inset",
        duration: 0.3,
        ease: "power2.out",
      });

      const onMouseEnter = () => hoverTimeline.play();
      const onMouseLeave = () => hoverTimeline.reverse();

      element.addEventListener("mouseenter", onMouseEnter);
      element.addEventListener("mouseleave", onMouseLeave);

      // Store event listeners for cleanup
      element._listeners = { onMouseEnter, onMouseLeave };
    });

    // Cleanup
    return () => {
      Array.from(currentItemRefs.entries()).forEach(([_id, element]) => {
        if (element && element._listeners) {
          const { onMouseEnter, onMouseLeave } = element._listeners;
          element.removeEventListener("mouseenter", onMouseEnter);
          element.removeEventListener("mouseleave", onMouseLeave);
          gsap.killTweensOf(element);
        }
      });
    };
  }, [items]);

  return (
    <div ref={listRef} className={`space-y-3 ${className}`}>
      {items.map((item) => (
        <div
          key={item.id}
          ref={(el) => {
            if (el) itemRefs.current.set(item.id, el);
            else itemRefs.current.delete(item.id);
          }}
          className="transition-all duration-300 ease-in-out"
        >
          {renderItem(item)}
        </div>
      ))}
    </div>
  );
}
