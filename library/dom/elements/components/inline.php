<?php

/**
 * Description of inline
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\components\intext;

class inline extends intext{

    
    public function __construct($tag = "span",$value=false) {
        parent::__construct($tag,$value);
    }


}
