export const USD_TO_INR = 83;

/**
 * Format a USD amount into Indian Rupees (INR) with standard Indian numbering format.
 * @param {number} amountInUSD
 * @param {object} options
 * @returns {string} e.g. "₹2,71,06,040"
 */
export function formatINR(amountInUSD, { showDecimals = false } = {}) {
  if (amountInUSD === null || amountInUSD === undefined || isNaN(amountInUSD)) {
    return '₹0';
  }
  const inr = Number(amountInUSD) * USD_TO_INR;
  if (showDecimals) {
    return '₹' + inr.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return '₹' + Math.round(inr).toLocaleString('en-IN');
}

/**
 * Format an amount that is already in Indian Rupees.
 * @param {number} amountInINR
 * @param {object} options
 * @returns {string} e.g. "₹7,500"
 */
export function formatINRDirect(amountInINR, { showDecimals = false } = {}) {
  if (amountInINR === null || amountInINR === undefined || isNaN(amountInINR)) {
    return '₹0';
  }
  const val = Number(amountInINR);
  if (showDecimals) {
    return '₹' + val.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  return '₹' + Math.round(val).toLocaleString('en-IN');
}
