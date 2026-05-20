import { redirect } from "next/navigation";

export default function VariantRoot({ params }: { params: { variant: string } }) {
  redirect(`/proto/${params.variant}/dashboard`);
}
