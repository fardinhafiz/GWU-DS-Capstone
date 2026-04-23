const API_BASE = 'http://127.0.0.1:8000'

async function parseJson(response) {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || 'Request failed')
  }
  return response.json()
}

export async function getHealth() {
  return parseJson(await fetch(`${API_BASE}/health`))
}

export async function fetchOptions() {
  return parseJson(await fetch(`${API_BASE}/options`))
}

export async function fetchPreferences(userId) {
  return parseJson(await fetch(`${API_BASE}/preferences/${userId}`))
}

export async function saveProfile(userId, payload) {
  return parseJson(
    await fetch(`${API_BASE}/preferences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        gender: payload.gender ?? 'All',
        preferred_colors: payload.preferred_colors ?? [],
        disliked_colors: payload.disliked_colors ?? [],
        preferred_categories: payload.preferred_categories ?? [],
        preferred_types: payload.preferred_types ?? [],
        preferred_usage: payload.preferred_usage ?? [],
      }),
    })
  )
}

export async function searchCatalog(userId, query) {
  return parseJson(
    await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, query }),
    })
  )
}

export async function recommend(userId) {
  return parseJson(await fetch(`${API_BASE}/recommend/${userId}`))
}

export async function likeItem(userId, itemId) {
  return parseJson(
    await fetch(`${API_BASE}/like`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, item_id: itemId }),
    })
  )
}

export async function dislikeItem(userId, itemId) {
  return parseJson(
    await fetch(`${API_BASE}/dislike`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, item_id: itemId }),
    })
  )
}

export async function uploadStyleImage(userId, file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/style-image/${userId}`, {
    method: 'POST',
    body: formData,
  })

  return parseJson(response)
}

export async function fetchStyleProfile(userId) {
  return parseJson(await fetch(`${API_BASE}/style-profile/${userId}`))
}

export function toImageUrl(item) {
  if (!item?.id) return null
  return `${API_BASE}/images/${item.id}.jpg`
}

export function toUploadedStyleImageUrl(path) {
  if (!path) return null
  return `${API_BASE}${path}`
}
