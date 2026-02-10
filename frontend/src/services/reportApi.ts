const API_BASE = '/api/v1/reports';

/** Extract filename from Content-Disposition header and trigger browser download. */
function downloadBlob(blob: Blob, systemId: string, response: Response): void {
  const disposition = response.headers.get('Content-Disposition') || '';
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `UAIE_Report_${systemId.slice(0, 8)}.pdf`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const reportApi = {
  /**
   * Download a PDF report for the given system.
   * Triggers a browser file download.
   */
  async downloadReport(systemId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/systems/${systemId}/pdf`);
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Download failed' }));
      throw new Error(err.detail || 'Failed to download report');
    }
    const blob = await response.blob();
    downloadBlob(blob, systemId, response);
  },

  /**
   * Run a fresh analysis and immediately download the PDF report.
   */
  async analyzeAndDownload(systemId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/systems/${systemId}/analyze-and-report`, {
      method: 'POST',
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Report generation failed' }));
      throw new Error(err.detail || 'Failed to generate report');
    }
    const blob = await response.blob();
    downloadBlob(blob, systemId, response);
  },
};
