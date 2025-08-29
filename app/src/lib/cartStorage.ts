export type CartItem = {
  id: number; // Cart component expects number
  name: string;
  price: number;
  quantity: number;
  image: string;
  aiRecommended?: boolean;
  vendor?: string | null;
  url?: string | null;
  currency?: string | null;
};

const CART_KEY = "cartItems";

export const loadCart = (): CartItem[] => {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(CART_KEY);
    return raw ? (JSON.parse(raw) as CartItem[]) : [];
  } catch {
    return [];
  }
};

export const saveCart = (items: CartItem[]) => {
  if (typeof window === "undefined") return;
  localStorage.setItem(CART_KEY, JSON.stringify(items));
  // notify any listeners (Cart, header badge, etc.)
  window.dispatchEvent(new CustomEvent("cart:changed", { detail: items }));
};

export const addToCart = (newItem: CartItem) => {
  const items = loadCart();
  // de-dupe by product + vendor (so BestBuy vs Amazon are separate lines)
  const idx = items.findIndex(
    it =>
      it.name.toLowerCase() === newItem.name.toLowerCase() &&
      (it.vendor || "") === (newItem.vendor || "")
  );
  if (idx >= 0) {
    items[idx].quantity += newItem.quantity;
  } else {
    items.unshift(newItem);
  }
  saveCart(items);
  return items;
};
