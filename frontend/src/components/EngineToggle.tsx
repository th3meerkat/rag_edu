import type { Engine } from '../api'

type EngineToggleProps = {
  value: Engine
  onChange: (engine: Engine) => void
}

const ENGINES: Engine[] = ['langchain', 'llamaindex']

/**
 * Segmented control that switches between the available backend engines.
 */
export function EngineToggle({ value, onChange }: EngineToggleProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Backend engine"
      className="inline-flex rounded-full bg-slate-200 p-1 text-sm"
    >
      {ENGINES.map((engine) => {
        const active = engine === value
        return (
          <button
            key={engine}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(engine)}
            className={
              'rounded-full px-4 py-1 transition-colors ' +
              (active
                ? 'bg-white text-slate-900 shadow'
                : 'text-slate-600 hover:text-slate-900')
            }
          >
            {engine}
          </button>
        )
      })}
    </div>
  )
}
