<?php
/**
 * Description of link
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\intext;


class link extends intext{
    public function __construct($value = false, $href = "#") {
        parent::__construct("a", $value);
        $this->attributes["href"] = $href;
    }
}
