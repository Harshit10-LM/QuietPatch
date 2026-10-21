import React from 'react';

interface SlothMascotProps {
  className?: string;
  size?: number;
}

export const SlothMascot: React.FC<SlothMascotProps> = ({ className = 'w-7 h-7', size = 28 }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="QuietPatch Sloth Mascot"
    >
      {/* Branch */}
      <path
        d="M2 13C8 13 14 11 30 11"
        stroke="#475569"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      {/* Sloth Body hanging */}
      <ellipse cx="16" cy="18" rx="8" ry="6" fill="#10b981" fillOpacity="0.85" />
      {/* Hanging Arms */}
      <path
        d="M10 16C10 13 11 12 11 11"
        stroke="#34d399"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M22 16C22 13 21 12 21 11"
        stroke="#34d399"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Sloth Face Mask */}
      <circle cx="16" cy="17" r="4" fill="#064e3b" />
      {/* Eyes (calm, resting) */}
      <path d="M14 16.5C14.5 17 15 17 15.5 16.5" stroke="#a7f3d0" strokeWidth="1" strokeLinecap="round" />
      <path d="M16.5 16.5C17 17 17.5 17 18 16.5" stroke="#a7f3d0" strokeWidth="1" strokeLinecap="round" />
      {/* Friendly gentle smile */}
      <path d="M15 19C15.5 19.5 16.5 19.5 17 19" stroke="#6ee7b7" strokeWidth="1" strokeLinecap="round" />
    </svg>
  );
};
