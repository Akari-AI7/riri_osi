class FaceFeatureAnalyzer:
    def __init__(self):
        pass

    def generate_feature_descriptions(self, changes):
        """特徴変化の自然言語記述生成（輪郭・むくみ・対称性にも対応）"""
        descriptions = []

        for feature, data in changes.items():
            if isinstance(data, dict) and "change_percent" in data:
                change_percent = data["change_percent"]
                abs_percent = abs(change_percent)

                # 変化量に応じた表現
                if abs_percent < 5:
                    magnitude = "わずかに"
                elif abs_percent < 15:
                    magnitude = "やや"
                else:
                    magnitude = "顕著に"

                # 各特徴ごとの表現ルール
                if feature == "目":
                    direction = "大きくなっています" if change_percent > 0 else "小さくなっています"
                elif feature == "鼻":
                    direction = "長くなっています" if change_percent > 0 else "短くなっています"
                elif feature == "口":
                    direction = "横に広がっています" if change_percent > 0 else "横幅が狭まっています"
                elif feature == "眉":
                    direction = "上がっています" if change_percent > 0 else "下がっています"
                elif feature == "頬":
                    direction = "ふっくらしています" if change_percent > 0 else "すっきりしています"
                elif feature == "輪郭":
                    direction = "シャープになっています" if change_percent < 0 else "丸みを帯びています"
                elif feature == "むくみ":
                    direction = "増えています" if change_percent > 0 else "減っています"
                elif feature == "左右対称性":
                    direction = "改善しています" if change_percent > 0 else "低下しています"
                else:
                    direction = "変化があります"

                description = f"{feature}が{magnitude}{direction}"

                # 大きめの変化は数値を表示
                if abs_percent > 10:
                    description += f"（{change_percent:+.1f}%）"

                descriptions.append(description)

        if not descriptions:
            descriptions.append("顔の特徴に顕著な変化は見られませんでした。")

        # 総合的な洞察を追加
        self.add_insights(descriptions, changes)
        return descriptions

    def add_insights(self, descriptions, changes):
        """変化の洞察を追加"""
        if "頬" in changes and "むくみ" in changes:
            if changes["頬"]["change_percent"] > 0 and changes["むくみ"]["change_percent"] > 0:
                descriptions.append("全体的に顔がふっくらしており、むくみが目立ちます。")
            elif changes["頬"]["change_percent"] < 0 and changes["むくみ"]["change_percent"] < 0:
                descriptions.append("顔全体がすっきりしてシャープになっています。")

        if "左右対称性" in changes and changes["左右対称性"]["change_percent"] > 5:
            descriptions.append("顔のバランスが改善され、より整った印象になっています。")
