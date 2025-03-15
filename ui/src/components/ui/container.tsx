import * as React from "react"
import { cn } from "@/lib/utils"

interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Container({
  className,
  ...props
}: ContainerProps) {
  return (
    <div
      className={cn(
        "container mx-auto",
        className
      )}
      {...props}
    />
  )
}

interface WrapperProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Wrapper({
  className,
  ...props
}: WrapperProps) {
  return (
    <div
      className={cn(
        "px-6",
        className
      )}
      {...props}
    />
  )
} 