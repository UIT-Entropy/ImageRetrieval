import { useState, type ImgHTMLAttributes, type SyntheticEvent } from 'react'

const DEFAULT_MAX_RETRIES = 2
const FAILED_IMAGE = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360">
    <rect width="640" height="360" fill="#090d13"/>
    <path d="M270 130h100v76H270z M288 112h64l12 18h-88z" fill="none" stroke="#607086" stroke-width="8"/>
    <path d="m287 191 23-24 19 18 13-12 17 18" fill="none" stroke="#607086" stroke-width="7"/>
    <text x="320" y="245" fill="#8b9bb0" font-family="sans-serif" font-size="18" text-anchor="middle">Không tải được ảnh</text>
  </svg>
`)}`

const buildImageRetryUrl = (source: string, attempt: number): string | null => {
  try {
    const url = new URL(source, window.location.href)
    if (!url.pathname.startsWith('/media/keyframes/')) return null
    url.searchParams.set('refresh', 'true')
    url.searchParams.set('_retry', `${attempt}-${Date.now()}`)
    return url.toString()
  } catch {
    return null
  }
}

type RetryImageProps = Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> & {
  src: string
  maxRetries?: number
}

type ImageState = {
  originalSource: string
  displaySource: string
  attempt: number
  failed: boolean
}

const initialState = (source: string): ImageState => ({
  originalSource: source,
  displaySource: source,
  attempt: 0,
  failed: false,
})

export function RetryImage({ src, maxRetries = DEFAULT_MAX_RETRIES, onError, ...props }: RetryImageProps) {
  const [storedState, setStoredState] = useState(() => initialState(src))
  const state = storedState.originalSource === src ? storedState : initialState(src)

  const retry = (event: SyntheticEvent<HTMLImageElement>) => {
    const nextAttempt = state.attempt + 1
    const retrySource = nextAttempt <= maxRetries ? buildImageRetryUrl(src, nextAttempt) : null
    if (retrySource) {
      setStoredState({
        originalSource: src,
        displaySource: retrySource,
        attempt: nextAttempt,
        failed: false,
      })
      return
    }

    setStoredState({
      originalSource: src,
      displaySource: FAILED_IMAGE,
      attempt: state.attempt,
      failed: true,
    })
    onError?.(event)
  }

  return (
    <img
      {...props}
      src={state.displaySource}
      onError={retry}
      data-image-retries={state.attempt}
      data-image-failed={state.failed || undefined}
    />
  )
}
