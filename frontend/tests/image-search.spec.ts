import { expect, test } from '@playwright/test'
import path from 'node:path'

// Uses the real verified catalog. Start search_api.py on localhost:8100 first.
test.beforeEach(async ({ page }) => {
  await page.goto('/images')
  await page.getByRole('button', { name: 'Luyện tập', exact: true }).click()
  await page.getByRole('button', { name: /^Tìm kiếm/ }).click()
})

test('filters real images and inspects detected object masks', async ({ page }) => {
  await expect(page.getByText('DATA READY', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '+ Điều kiện', exact: true }).click()
  await page.getByLabel('Object 1', { exact: true }).selectOption('person')
  const response = page.waitForResponse(r => r.url().endsWith('/api/images/search') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Tìm ảnh', exact: true }).click()
  const result = await (await response).json()
  expect(result.total).toBeGreaterThan(0)
  expect(result.items.every((item: { objectCounts: Record<string, number> }) => item.objectCounts.person > 0)).toBeTruthy()
  await page.locator('.keyframe-tile').first().click()
  await expect(page.getByRole('button', { name: 'Tìm tương tự', exact: true })).toBeVisible()
  await page.locator('.image-object-row').first().click()
  const mask = page.locator('.image-mask-layer')
  await expect(mask).toBeVisible()
  await expect.poll(() => mask.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0)
  await expect(page.locator('.image-inspector')).toHaveCSS('width', /\d+px/)
})

test('searches by existing image vectors and keeps shortlist removable', async ({ page }) => {
  await page.locator('.keyframe-tile').first().click()
  await page.getByRole('button', { name: 'Tìm tương tự', exact: true }).click()
  await expect(page.getByText(/cosine similarity/)).toBeVisible()
  await expect(page.locator('.keyframe-tile .score-chip').first()).toHaveText('1.000')
  await page.locator('.keyframe-tile .tile-star').first().click()
  await page.getByRole('button', { name: 'Shortlist (1)', exact: true }).click()
  await expect(page.locator('.keyframe-tile')).toHaveCount(1)
  await page.locator('.keyframe-tile .tile-star').first().click()
  await expect(page.locator('.keyframe-tile')).toHaveCount(0)
})

test('encodes Vietnamese text and a new uploaded image through the live GPU', async ({ page }) => {
  test.skip(process.env.IMAGE_ENCODER_E2E !== '1', 'Requires a live authenticated Qwen encoder')
  await expect(page.getByRole('textbox', { name: 'Mô tả nội dung', exact: true })).toBeEnabled()
  await page.getByRole('textbox', { name: 'Mô tả nội dung', exact: true }).fill('Một người mặc áo đỏ đang đứng trong bếp.')
  const textResponse = page.waitForResponse(r => r.url().endsWith('/api/images/search') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Tìm ảnh', exact: true }).click()
  expect((await textResponse).status()).toBe(200)
  await expect(page.getByText(/cosine similarity/)).toBeVisible()
  await page.getByRole('textbox', { name: 'Mô tả nội dung', exact: true }).fill('')
  await page.locator('input[type=file]').setInputFiles(path.resolve('../data/Batch1/Batch1/08_1002.jpg'))
  await expect(page.getByAltText('Ảnh query', { exact: true })).toBeVisible()
  const imageResponse = page.waitForResponse(r => r.url().endsWith('/api/images/search') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Tìm ảnh', exact: true }).click()
  const result = await (await imageResponse).json()
  expect(result.ranked).toBeTruthy()
  expect(result.items[0].score).toBeGreaterThan(.9)
})
