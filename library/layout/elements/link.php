<?php
/**
 * Description of link
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;
class link extends element{
    public function __construct($value, $href = "#") {
        $this->value = $value;
        $this->attributes["href"] = $href;
        $this->tag = "a";
        $this->closeTag = true;
    }
}
