import { apiClient } from './client'

export interface HealthStatus {
  status: string
  env: string
}

export async function getHealth(): Promise<HealthStatus> {
  const response = await apiClient('/health')
  return (await response.json()) as HealthStatus
}
