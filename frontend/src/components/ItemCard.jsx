import { useState } from 'react'
import { toImageUrl } from '../api'

export default function ItemCard({ item, onFeedback, busy }) {
  const imageUrl = toImageUrl(item)
  const title = item.productDisplayName || `Item ${item.id}`
  const color = item.baseColour || 'Unknown color'
  const articleType = item.articleType || 'Unknown type'
  const category = item.masterCategory || 'Category'
  const usage = item.usage || ''
  const subcategory = item.subCategory || ''
  const semantic = Number(item.semantic_score || 0)
  const preference = Number(item.pref_score || 0)
  const xgbScore = Number(item.xgb_score ?? item.final_score ?? 0)

  const [selectedAction, setSelectedAction] = useState(null)

  async function handleClick(action) {
    try {
      await onFeedback(item.id, action)
      setSelectedAction(action)
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <article className="item-card">
      <div className="item-image-shell">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={title}
            className="item-image"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
              const fallback = e.currentTarget.parentElement?.querySelector('.image-fallback')
              if (fallback) fallback.style.display = 'block'
            }}
          />
        ) : null}
        <div className="image-fallback" style={{ display: imageUrl ? 'none' : 'block' }}>
          No image available
        </div>
      </div>

      <div className="item-body">
        <h3>{title}</h3>
        <p className="muted">
          {color} · {articleType}
        </p>

        <div className="pill-row">
          <span className="pill">{category}</span>
          {usage ? <span className="pill">{usage}</span> : null}
          {subcategory ? <span className="pill">{subcategory}</span> : null}
        </div>

        <div className="score-box">
          <span>Semantic {semantic.toFixed(2)}</span>
          <span>Preference {preference.toFixed(2)}</span>
          <span>XGBoost {xgbScore.toFixed(2)}</span>
        </div>

        <div className="action-row">
          <button
            className="primary-btn"
            disabled={busy}
            onClick={() => handleClick('like')}
            style={{
              opacity: selectedAction === 'dislike' ? 0.6 : 1,
              outline: selectedAction === 'like' ? '2px solid #22c55e' : 'none'
            }}
          >
            {selectedAction === 'like' ? '✅ Liked' : '👍 Like'}
          </button>

          <button
            className="ghost-btn"
            disabled={busy}
            onClick={() => handleClick('dislike')}
            style={{
              opacity: selectedAction === 'like' ? 0.6 : 1,
              outline: selectedAction === 'dislike' ? '2px solid #ef4444' : 'none'
            }}
          >
            {selectedAction === 'dislike' ? '🚫 Disliked' : '👎 Dislike'}
          </button>
        </div>
      </div>
    </article>
  )
}
