# Leaderboard

Time-of-day cross-validation (7 rounds, 2 photos per camera held out each), official points on all 1962 labelled areas. Baseline (always compliant + always full): +1598 of 3924. Group C references are rebuilt per round without the held-out photos; runs marked @shared used references built from all photos (slightly optimistic); runs marked @camera hold out one whole camera per round instead (14 rounds, shared references).

| date | run | total | vs baseline | compliance | stock | most-likely rule |
|---|---|---|---|---|---|---|
| 2026-10-10 22:37 | `logreg_handcrafted@shared` | +2744 | +1146 | +1695 | +1049 | +2681 |
| 2026-10-10 22:37 | `logreg_siglip2@shared` | +2727 | +1129 | +1695 | +1032 | +2703 |
| 2026-10-10 22:37 | `logreg_dinov2@shared` | +2813 | +1215 | +1743 | +1070 | +2819 |
| 2026-10-10 22:37 | `logreg_dinov3@shared` | +3003 | +1405 | +1722 | +1281 | +3013 |
| 2026-10-10 22:37 | `logreg_radio@shared` | +3033 | +1435 | +1722 | +1311 | +3010 |
| 2026-10-10 22:37 | `logreg_handcrafted+siglip2@shared` | +3186 | +1588 | +1845 | +1341 | +3180 |
| 2026-10-10 22:37 | `logreg_handcrafted+dinov2@shared` | +3115 | +1517 | +1830 | +1285 | +3140 |
| 2026-10-10 22:37 | `logreg_handcrafted+dinov3@shared` | +3215 | +1617 | +1797 | +1418 | +3209 |
| 2026-10-10 22:38 | `logreg_handcrafted+radio@shared` | +3288 | +1690 | +1830 | +1458 | +3276 |
| 2026-10-10 22:38 | `logreg_handcrafted+siglip2+dinov3@shared` | +3335 | +1737 | +1857 | +1478 | +3330 |
| 2026-10-10 22:38 | `hgb_handcrafted@shared` | +3422 | +1824 | +1884 | +1538 | +3426 |
| 2026-10-10 22:38 | `hgb_handcrafted+siglip2@shared` | +3448 | +1850 | +1878 | +1570 | +3453 |
| 2026-10-10 22:38 | `hgb_handcrafted+dinov3@shared` | +3473 | +1875 | +1902 | +1571 | +3452 |
| 2026-10-10 22:38 | `knn_siglip2@shared` | +2795 | +1197 | +1863 | +932 | +2781 |
| 2026-10-10 22:38 | `knn_dinov3@shared` | +2908 | +1310 | +1884 | +1024 | +2814 |
| 2026-10-10 22:38 | `knn_radio@shared` | +2789 | +1191 | +1857 | +932 | +2751 |
| 2026-10-10 22:39 | `hgb_handcrafted+radio@shared` | +3498 | +1900 | +1887 | +1611 | +3500 |
| 2026-10-10 22:39 | `hgb_handcrafted+siglip2+dinov3@shared` | +3462 | +1864 | +1887 | +1575 | +3452 |
| 2026-10-10 22:50 | `logreg_handcrafted` | +2672 | +1074 | +1698 | +974 | +2674 |
| 2026-10-10 22:50 | `logreg_handcrafted+siglip2+dinov3` | +3296 | +1698 | +1824 | +1472 | +3296 |
| 2026-10-10 22:51 | `hgb_handcrafted` | +3450 | +1852 | +1899 | +1551 | +3440 |
| 2026-10-10 22:51 | `hgb_handcrafted+siglip2` | +3477 | +1879 | +1893 | +1584 | +3452 |
| 2026-10-10 22:51 | `hgb_handcrafted+dinov3` | +3522 | +1924 | +1905 | +1617 | +3499 |
| 2026-10-10 22:51 | `hgb_handcrafted+radio` | +3538 | +1940 | +1896 | +1642 | +3533 |
| 2026-10-10 22:52 | `hgb_handcrafted+siglip2+dinov3` | +3514 | +1916 | +1899 | +1615 | +3481 |
| 2026-10-11 01:13 | `hgb_handcrafted+radio+proto_dinov2+proto_radio+proto_siglip2` | +3489 | +1891 | +1884 | +1605 | +3502 |
| 2026-10-11 01:14 | `hgb_handcrafted+radio+proto_dinov2+proto_radio+proto_siglip2+override-proto` | +3489 | +1891 | +1884 | +1605 | +3502 |
| 2026-10-11 01:14 | `hgb_handcrafted+radio+override-proto` | +3543 | +1945 | +1914 | +1629 | +3539 |
| 2026-10-11 01:14 | `hgb_handcrafted+radio+override-proto-only` | +3489 | +1891 | +1860 | +1629 | +3485 |
| 2026-10-11 01:15 | `hgb_handcrafted+radio+override-text` | +3501 | +1903 | +1872 | +1629 | +3497 |
| 2026-10-11 01:15 | `hgb_handcrafted+radio@camera` | +2484 | +886 | +1560 | +924 | +2472 |
| 2026-10-11 01:16 | `hgb_handcrafted+radio+override-proto@camera` | +2436 | +838 | +1512 | +924 | +2424 |
| 2026-10-11 01:17 | `hgb_handcrafted+radio+override-proto-only@camera` | +2436 | +838 | +1512 | +924 | +2424 |
| 2026-10-11 01:19 | `hgb_handcrafted+radio+override-proto-3votes` | +3552 | +1954 | +1923 | +1629 | +3548 |
