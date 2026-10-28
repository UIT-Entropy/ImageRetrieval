import type { ButtonHTMLAttributes } from 'react'
export const Button = ({ className = '', children, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) => <button className={`btn ${className}`} {...props}>{children}</button>
export const Spinner = () => <span className="spinner" aria-label="Đang tải" />
